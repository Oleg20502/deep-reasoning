import torch
from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
from modeling_rmt.language_modeling import RecurrentWrapper


# from transformers import AutoTokenizer
# tokenizer = AutoTokenizer.from_pretrained('gpt2')


class RecurrentWrapperNoSegmentation(RecurrentWrapper):
    def forward(self, segments, labels, output_attentions=None, output_hidden_states=None):
        # segments = segments['segments']
        # for seg in segments:
        #     print(tokenizer.batch_decode(seg['input_ids'], skip_special_tokens=True))
        memory_state = None

        cell_outputs = []
        for seg_num, segment in enumerate(segments):
            cell_out, memory_state = self.memory_cell(input_ids=segment['input_ids'],
                                                      attention_mask=segment['attention_mask'],
                                                      memory_state=memory_state, output_hidden_states=True)
            cell_outputs.append(cell_out)
            self.manage_gradients(memory_state, seg_num)

        out = self.process_outputs(cell_outputs, segments,
                                   output_attentions=output_attentions,
                                   output_hidden_states=output_hidden_states)
        return out

    def generate(self, segments, **generate_kwargs):
        raise NotImplementedError("Generation not implemented for this wrapper.")
        memory_state = None
        for seg_num, segment in enumerate(segments[:-1]):
            cell_out, memory_state = self.memory_cell(**segment, memory_state=memory_state, output_hidden_states=True)

        final_segment = segments[-1]
        out = self.memory_cell.generate(**final_segment, memory_state=memory_state, **generate_kwargs)

        return out

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

                loss_fct = CrossEntropyLoss()
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
        # print(out.keys(), out.loss)

        return out

    def gradient_checkpointing_enable(self, *args, **kwargs):
        if hasattr(self.memory_cell.model, "gradient_checkpointing_enable"):
            return self.memory_cell.model.gradient_checkpointing_enable(*args, **kwargs)


class RecurrentWrapperNoSegmentationWeighLoss(RecurrentWrapperNoSegmentation):
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

                loss_fct = CrossEntropyLoss()
                labels_mask = segment.get('labels_mask')
                if labels_mask is not None:
                    shift_mask = labels_mask[..., :-1].contiguous()

                    flat_labels = flat_labels[shift_mask.view(-1)]
                    flat_logits = flat_logits[shift_mask.view(-1)]

                    if labels_mask.sum() == 0:
                        loss_value = 0
                    else:
                        loss_value = loss_fct(flat_logits, flat_labels)

                if seg_num == len(segments) - 1:
                    answer_weight = self.rmt_config.get("answer_loss_weight", 1)
                    loss_value *= answer_weight

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
        # print(out.keys(), out.loss)

        return out

    def gradient_checkpointing_enable(self, *args, **kwargs):
        if hasattr(self.memory_cell.model, "gradient_checkpointing_enable"):
            return self.memory_cell.model.gradient_checkpointing_enable(*args, **kwargs)
