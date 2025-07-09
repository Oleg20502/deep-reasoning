import torch
from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions

from modeling_rmt.language_modeling import *
from modeling_amt.language_modeling import AssociativeRecurrentWrapper


class AssociativeRecurrentWrapperNoSegmentation(AssociativeRecurrentWrapper):
    def forward(self, segments, labels,
                output_attentions=None,
                output_hidden_states=None,
                sliding_window=False):
        cell_outputs = []
        past_key_values = None
        num_mem_tokens = self.memory_cell.num_mem_tokens
        prev_attn_mask = None
        self.memory_cell.zero_mem()
        for seg_num, segment in enumerate(segments):
            seg_len = segment['input_ids'].size(-1)
            cell_out = self.memory_cell(input_ids=segment['input_ids'],
                                        attention_mask=segment['attention_mask'],
                                        labels=segment['labels'],
                                        labels_mask=segment['labels_mask'],
                                        output_hidden_states=True,
                                        use_cache=sliding_window,
                                        past_key_values=past_key_values,
                                        prev_attn_mask=prev_attn_mask,
                                        zero_mem=False
                                        )
            if sliding_window:
                prev_attn_mask = segment['attention_mask']
                past_key_values = [
                    [
                        k_or_v[..., -(num_mem_tokens+seg_len):k_or_v.size(-2)-num_mem_tokens, :].detach()
                        for k_or_v in seg_kv
                    ]
                    for seg_kv in cell_out['past_key_values']
                ]
            cell_outputs.append(cell_out)
        self.memory_cell.zero_mem()

        out = self.process_outputs(cell_outputs, segments,
                                   output_attentions=output_attentions,
                                   output_hidden_states=output_hidden_states)
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

            if self.rmt_config.get("return_all_logits", False):
                out['ce_loss'] = out['loss']

            segment_keys = ['loss']
            if kwargs.get('output_attentions'):
                segment_keys.append('attentions')
            if kwargs.get('output_hidden_states'):
                segment_keys.append('hidden_states')

            if self.rmt_config.get("return_all_logits", False):
                for seg_num, o in enumerate(cell_outputs):
                    for key, value in o.items():
                        if any([sk in key for sk in segment_keys]):
                            proxy_out[f'{key}_{seg_num}'] = value

        num_segments = len(segments)
        out['loss'] = sum([proxy_out[f'loss_{seg_num}'] for seg_num in range(num_segments)]) / num_segments
        out['logits'] = torch.cat([cell_out.logits for cell_out in cell_outputs], dim=1)
        return out


class Distillator(torch.nn.Module):
    def __init__(self, teacher_model, student_model, alpha_distil):
        super().__init__()
        self.teacher = teacher_model
        self.student = student_model
        self.alpha = alpha_distil
        for p in self.teacher.parameters():
            p.requires_grad = False
    
    def forward(self, 
                input_ids, 
                labels=None, 
                labels_mask=None, 
                inputs_embeds=None, 
                attention_mask=None, 
                output_attentions=None, 
                output_hidden_states=None,
                input_segmented=False,
                sliding_window=False,
                ):
        with torch.no_grad():
            if self.training:
                if input_segmented:
                    raise NotImplementedError
                    n_segs = input_ids.shape[1] if not (input_ids is None) else inputs_embeds.shape[1]
                    teacher_output = self.teacher(
                        input_ids=torch.cat([input_ids[:, i] for i in range(n_segs)], dim=1) if input_ids is not None else None,
                        labels=torch.cat([labels[:, i] for i in range(n_segs)], dim=1)  if labels is not None else None, 
                        inputs_embeds=torch.cat([inputs_embeds[:, i] for i in range(n_segs)], dim=1) if inputs_embeds is not None else None, 
                        attention_mask=torch.cat([attention_mask[:, i] for i in range(n_segs)], dim=1) if attention_mask is not None else None, 
                        output_attentions=output_attentions, 
                        output_hidden_states=output_hidden_states
                    )
                else:
                    teacher_output = self.teacher(
                        input_ids=input_ids,
                        labels=labels, 
                        inputs_embeds=inputs_embeds, 
                        attention_mask=attention_mask, 
                        output_attentions=output_attentions, 
                        output_hidden_states=output_hidden_states
                    )
            else: 
                teacher_output = dict()
        student_output = self.student(
            input_ids=input_ids,
            labels=labels,
            labels_mask=labels_mask, 
            inputs_embeds=inputs_embeds, 
            attention_mask=attention_mask, 
            output_attentions=output_attentions, 
            output_hidden_states=output_hidden_states,
            input_segmented=input_segmented,
            sliding_window=sliding_window
        )
        
        if input_segmented:
            n_segs = input_ids.shape[1] if not (input_ids is None) else inputs_embeds.shape[1]
            labels = torch.cat([labels[:, i] for i in range(n_segs)], dim=1)
            if labels_mask is not None:
                labels_mask = torch.cat([labels_mask[:, i] for i in range(n_segs)], dim=1)
    
        out = self.process_outputs(teacher_output, student_output,
                                   labels=labels,
                                   labels_mask=labels_mask,
                                   output_attentions=output_attentions,
                                   output_hidden_states=output_hidden_states)

        return out
                  
    def process_outputs(self, teacher_output, student_output, **kwargs):
        out = CausalLMOutputWithCrossAttentions()
        teacher_logits = teacher_output.logits if self.training else None
        student_logits = student_output.logits

        for (k, v) in student_output.items():
            out[k] = v
            
            teachers = teacher_output.get(k)
            # if teachers is not None:
            #     out[f'teacher_{k}'] = teachers

        labels = kwargs.get('labels')
        if labels is not None:
            shift_labels = labels[..., 1:].contiguous()
            shift_logits = student_logits[..., :-1, :].contiguous()
            shift_t_logits = teacher_logits[..., :-1, :].contiguous() if self.training else None

            flat_labels = shift_labels.view(-1)
            flat_logits = shift_logits.view(-1, shift_logits.size(-1))
            flat_t_logits = shift_t_logits.view(-1, shift_t_logits.size(-1)) if self.training else None
            
            labels_mask = kwargs.get('labels_mask')
            if labels_mask is not None:
                shift_mask = labels_mask[..., :-1].contiguous()

                flat_labels = flat_labels[shift_mask.view(-1)]
                flat_logits = flat_logits[shift_mask.view(-1)]
                flat_t_logits = flat_t_logits[shift_mask.view(-1)] if self.training else None

            dist_fct = torch.nn.KLDivLoss(reduction='batchmean', log_target=True)

            log_sftmx_student = torch.log_softmax(flat_logits, dim=-1)  
            log_sftmx_teacher = torch.log_softmax(flat_t_logits, dim=-1) if self.training else None
            dist = dist_fct(log_sftmx_student, log_sftmx_teacher) if self.training else None
            # out['ce_loss'] = out['loss']
            ce_loss = out['loss']
            if self.training:
                # out['dist'] = dist
                # out['loss'] = (1 - self.alpha) * out['ce_loss'] + self.alpha * dist
                out['loss'] = (1 - self.alpha) * ce_loss + self.alpha * dist
            
        else:
            out['loss'] = 0

        return out 

