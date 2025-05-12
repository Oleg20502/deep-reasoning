import torch
import torch.nn as nn
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
from transformers.models.gpt2.modeling_gpt2 import GPT2Attention, GPT2MLP

from modeling_rmt.experimental import RecurrentWrapperNoSegmentationGenerate


class RMCrossAttention(torch.nn.Module):
    def __init__(self, config, dropout=False, ff=False):
        super().__init__()
        self.config = config
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        hidden_size = config.hidden_size
        self.ln = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.cross_attn = GPT2Attention(config=config, is_cross_attention=True)
        self.dropout = nn.Dropout(config.attention_dropout) if dropout else None

        inner_dim = config.n_inner if config.n_inner is not None else 4 * hidden_size
        self.ff = GPT2MLP(inner_dim, config) if ff else None

    def forward(self, hidden_states, encoder_hidden_states, attention_mask=None,
                output_attentions=False
                ):
        residual = hidden_states
        hidden_states = self.ln(hidden_states)
        cross_attn_outputs = self.cross_attn(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
        )
        attn_output = cross_attn_outputs[0]
        if self.dropout is not None:
            attn_output = self.dropout(attn_output)
        if self.ff is not None:
            attn_output = self.ff(attn_output)
        hidden_states = residual + attn_output

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (cross_attn_outputs[1:],)

        return outputs  # hidden_states, attention_weights


class RMCALayerWrapper(torch.nn.Module):
    def __init__(self, layer, num_mem_tokens, config):
        super().__init__()
        self.layer = layer
        self.create_memory(num_mem_tokens, config.hidden_size)
        self.memory_state = None
        self.mem_read_layer = self.create_mem_read_layer(config)
        self.mem_write_layer = self.create_mem_write_layer(config)
        self.generate_mode = False

    def forward(self, hidden_states, output_attentions=False, *args, **kwargs):
        # if args:
        #     print('wrapper args', args)
        # if kwargs:
        #     print('wrapper kwargs', kwargs)
        if self.memory_state is None:
            self.memory_state = self.set_memory(hidden_states)

        mem_read_out = self.mem_read_layer(hidden_states, encoder_hidden_states=self.memory_state,
                                           output_attentions=output_attentions)
        hidden_states, cross_attentions_read = mem_read_out[0], mem_read_out[1:]

        mem_write_out = self.mem_write_layer(self.memory_state, encoder_hidden_states=hidden_states,
                                             output_attentions=output_attentions)
        self.memory_state, cross_attentions_write = mem_write_out[0], mem_write_out[1:]

        out = self.layer(hidden_states, **kwargs, output_attentions=output_attentions)

        self.debug_state = dict(hidden_states=hidden_states, memory_state=self.memory_state, output=out,
                                cross_attentions_read=cross_attentions_read,
                                cross_attentions_write=cross_attentions_write)

        return out

    def create_mem_read_layer(self, config):
        mem_read_layer = RMCrossAttention(config, dropout=getattr(self, 'use_dropout', False))
        self.register_module("mem_read_layer", mem_read_layer)
        return mem_read_layer

    def create_mem_write_layer(self, config):
        mem_write_layer = RMCrossAttention(config, dropout=getattr(self, 'use_dropout', False))
        self.register_module("mem_write_layer", mem_write_layer)
        return mem_write_layer

    def create_memory(self, num_mem_tokens, memory_dim):
        self.num_mem_tokens = num_mem_tokens
        memory_weights = torch.randn((num_mem_tokens, memory_dim))
        self.register_parameter('memory', torch.nn.Parameter(memory_weights, requires_grad=True))

    def set_memory(self, hidden_states):
        memory = self.memory.repeat(hidden_states.shape[0], 1, 1)
        return memory

    def reset_memory(self):
        self.memory_state = None

    def detach_memory_state(self):
        if self.memory_state is not None:
            self.memory_state = self.memory_state.detach()


class RMCAMemoryCell(torch.nn.Module):
    def __init__(self, base_model, num_mem_tokens, layers_attr: str = 'transformer.h'):
        super().__init__()
        self.model = base_model
        self.num_mem_tokens = num_mem_tokens

        self.layers = self.model
        self.layers_attrs = layers_attr.split('.')
        for i, attr in enumerate(self.layers_attrs):
            self.layers = getattr(self.layers, attr)

        for i in range(len(self.layers)):
            self.layers[i] = RMCALayerWrapper(
                self.layers[i],
                self.num_mem_tokens,
                self.model.config
            )

    def reset_memory(self):
        for i in range(len(self.layers)):
            self.layers[i].reset_memory()

    def detach_memory_state(self):
        for i in range(len(self.layers)):
            self.layers[i].detach_memory_state()

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def generate(self, input_ids, memory_state, attention_mask=None, **generate_kwargs):
        raise NotImplementedError("Generation is not supported yet.")


class RMCAWrapperNoSegmentationGenerate(RecurrentWrapperNoSegmentationGenerate):
    def forward(self, segments, labels, output_attentions=None, output_hidden_states=None, *args, **kwargs):
        # if args:
        #     print('wrapper args', args)
        # if kwargs:
        #     print('wrapper kwargs', kwargs)
        self.memory_cell.reset_memory()

        cell_outputs = []
        for seg_num, segment in enumerate(segments):
            cell_out = self.memory_cell(input_ids=segment['input_ids'],
                                        attention_mask=segment['attention_mask'],
                                        output_attentions=output_attentions,
                                        output_hidden_states=True)
            cell_outputs.append(cell_out)
            self.manage_gradients(seg_num)

        out = self.process_outputs(cell_outputs, segments,
                                   output_attentions=output_attentions,
                                   output_hidden_states=output_hidden_states)
        return out

    def generate(self, segments, **kwargs):
        raise NotImplementedError("Generation is not supported yet.")
        memory_state = None

        for seg_num, segment in enumerate(segments):
            cell_out, memory_state = self.memory_cell(input_ids=segment['input_ids'],
                                                      attention_mask=segment['attention_mask'],
                                                      memory_state=memory_state, output_hidden_states=True)

        generated_segments = []
        for seg_num in range(len(segments), self.rmt_config.get("max_n_segments", 32)):
            output_ids, memory_state = self.generate_segment(memory_state=memory_state, **kwargs)
            generated_segments.append(output_ids)

            if self.all_done(generated_segments):
                break

        return generated_segments

    def manage_gradients(self, seg_num):
        k2, max_n_segments = self.rmt_config.get('k2', -1), self.rmt_config.get('max_n_segments')
        if seg_num == 0 \
            or k2 in {-1, None} \
                or seg_num + k2 > max_n_segments:
            return

        self.memory_cell.detach_memory_state()

    def process_outputs(self, cell_outputs, segments, **kwargs):
        out = CausalLMOutputWithCrossAttentions()
        proxy_out = {}
        for seg_num, segment in enumerate(segments):
            cell_out = cell_outputs[seg_num]

            full_logits = cell_out.logits

            labels = segment.get('labels')
            if labels is not None:
                shift_labels = labels[..., 1:].contiguous()
                shift_logits = full_logits[..., :-1, :].contiguous()
                flat_labels = shift_labels.view(-1)
                flat_logits = shift_logits.view(-1, shift_logits.size(-1))

                loss_fct = nn.CrossEntropyLoss()
                labels_mask = segment.get('labels_mask')
                if labels_mask is not None:
                    shift_mask = labels_mask[..., :-1].contiguous()

                    flat_labels = flat_labels[shift_mask.view(-1)]
                    flat_logits = flat_logits[shift_mask.view(-1)]

                    if labels_mask.sum() == 0:
                        loss_value = 0
                    else:
                        loss_value = loss_fct(flat_logits, flat_labels)

                proxy_out[f'loss_{seg_num}'] = loss_value
            else:
                proxy_out[f'loss_{seg_num}'] = 0

            segment_keys = ['loss']
            if kwargs.get('output_attentions'):
                segment_keys.append('attentions')
            if kwargs.get('output_hidden_states'):
                segment_keys.append('hidden_states')

            for key, value in cell_out.items():
                if any([sk in key for sk in segment_keys]):
                    proxy_out[f'{key}_{seg_num}'] = value

        num_segments = len(segments)
        out['loss'] = sum([proxy_out[f'loss_{seg_num}'] for seg_num in range(num_segments)]) / num_segments
        out['logits'] = torch.cat([cell_out.logits for cell_out in cell_outputs], dim=1)

        return out
