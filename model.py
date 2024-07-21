# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import torch.nn as nn
import torch    
import torch.nn.functional as F
from transformers import BartForConditionalGeneration, BartConfig
from transformers.models.bart.modeling_bart import BartClassificationHead

class RobertaLMHead(nn.Module):
    """Roberta Head for masked language modeling."""

    def __init__(self, input_dim, output_dim = 1):
        super().__init__()
        self.dense = nn.Linear(input_dim, input_dim)
        self.layer_norm = nn.LayerNorm(input_dim)

        self.decoder = nn.Linear(input_dim, output_dim)
        self.bias = nn.Parameter(torch.zeros(output_dim))
        self.decoder.bias = self.bias

    def forward(self, features, **kwargs):
        x = self.dense(features)
        x = nn.functional.gelu(x)
        x = self.layer_norm(x)

        # project back to size of vocabulary with bias
        x = self.decoder(x)
        return x

class CodeBERT(nn.Module):   
    def __init__(self, encoder):
        super(CodeBERT, self).__init__()
        self.encoder = encoder
    
    def forward(self, code_inputs=None, nl_inputs=None): 
        if code_inputs is not None:
            return self.encoder(code_inputs,attention_mask=code_inputs.ne(1))[1]
        else:
            return self.encoder(nl_inputs,attention_mask=nl_inputs.ne(1))[1]

class CodeBERT_RR(nn.Module): 
    def __init__(self, encoder):
        super(CodeBERT_RR, self).__init__()
        self.encoder = encoder
        hidden_size = self.encoder.pooler.dense.out_features
        self.head = RobertaLMHead(hidden_size, 1)
    
    def forward(self, inputs): 
        outputs = self.encoder(inputs,attention_mask=inputs.ne(1))[1]
        return torch.tanh(self.head(outputs))


class GraphCodeBERT(nn.Module):   
    def __init__(self, encoder):
        super(GraphCodeBERT, self).__init__()
        self.encoder = encoder
      
    def forward(self, code_inputs=None, attn_mask=None,position_idx=None, nl_inputs=None): 
        if code_inputs is not None:
            nodes_mask=position_idx.eq(0)
            token_mask=position_idx.ge(2)        
            inputs_embeddings=self.encoder.embeddings.word_embeddings(code_inputs)
            nodes_to_token_mask=nodes_mask[:,:,None]&token_mask[:,None,:]&attn_mask
            nodes_to_token_mask=nodes_to_token_mask/(nodes_to_token_mask.sum(-1)+1e-10)[:,:,None]
            avg_embeddings=torch.einsum("abc,acd->abd",nodes_to_token_mask,inputs_embeddings)
            inputs_embeddings=inputs_embeddings*(~nodes_mask)[:,:,None]+avg_embeddings*nodes_mask[:,:,None]    
            return self.encoder(inputs_embeds=inputs_embeddings,attention_mask=attn_mask,position_ids=position_idx)[1]
        else:
            return self.encoder(nl_inputs,attention_mask=nl_inputs.ne(1))[1]

class GraphCodeBERT_RR(nn.Module):   
    def __init__(self, encoder):
        super(GraphCodeBERT_RR, self).__init__()
        self.encoder = encoder
        hidden_size = self.encoder.pooler.dense.out_features
        self.head = RobertaLMHead(hidden_size, 1)
      
    def forward(self, inputs=None, attn_mask=None,position_idx=None): 
        if attn_mask is not None: 
            nodes_mask=position_idx.eq(0)
            token_mask=position_idx.ge(2)        
            inputs_embeddings=self.encoder.embeddings.word_embeddings(inputs)
            nodes_to_token_mask=nodes_mask[:,:,None]&token_mask[:,None,:]&attn_mask
            nodes_to_token_mask=nodes_to_token_mask/(nodes_to_token_mask.sum(-1)+1e-10)[:,:,None]
            avg_embeddings=torch.einsum("abc,acd->abd",nodes_to_token_mask,inputs_embeddings)
            inputs_embeddings=inputs_embeddings*(~nodes_mask)[:,:,None]+avg_embeddings*nodes_mask[:,:,None]    
            outputs = self.encoder(inputs_embeds=inputs_embeddings,attention_mask=attn_mask,position_ids=position_idx)[1]
        else: 
            outputs = self.encoder(inputs,attention_mask=inputs.ne(1))[1]
        return torch.tanh(self.head(outputs))

class Unixcoder(nn.Module):   
    def __init__(self, encoder):
        super(Unixcoder, self).__init__()
        self.encoder = encoder
    
    def forward(self, code_inputs=None, nl_inputs=None): 
        if code_inputs is not None:
            outputs = self.encoder(code_inputs,attention_mask=code_inputs.ne(1))[0]
            outputs = (outputs*code_inputs.ne(1)[:,:,None]).sum(1)/code_inputs.ne(1).sum(-1)[:,None]
            return torch.nn.functional.normalize(outputs, p=2, dim=1)
        else:
            outputs = self.encoder(nl_inputs,attention_mask=nl_inputs.ne(1))[0]
            outputs = (outputs*nl_inputs.ne(1)[:,:,None]).sum(1)/nl_inputs.ne(1).sum(-1)[:,None]
            return torch.nn.functional.normalize(outputs, p=2, dim=1)
        

class Unixcoder_RR(nn.Module): 
    def __init__(self, encoder): # , cosqa_flag = 0, query_len = 128
        super(Unixcoder_RR, self).__init__()
        self.encoder = encoder
        hidden_size = self.encoder.pooler.dense.out_features
        self.decoder = nn.Linear(hidden_size, 1)
        self.head = RobertaLMHead(hidden_size, 1)
    
    def forward(self, inputs): 
        outputs = self.encoder(inputs,attention_mask=inputs.ne(1))[0]
        outputs = (outputs*inputs.ne(1)[:,:,None]).sum(1)/inputs.ne(1).sum(-1)[:,None]
        
        return torch.tanh(self.head(outputs))


class BartForClassificationAndGeneration(BartForConditionalGeneration):

    def __init__(self, config: BartConfig, mode=None):
        super(BartForClassificationAndGeneration, self).__init__(config)
        
        # classification head
        self.classification_head = BartClassificationHead(
            config.d_model,
            config.d_model,
            config.num_labels,
            config.classifier_dropout,
        )
        self.model._init_weights(self.classification_head.dense)
        self.model._init_weights(self.classification_head.out_proj)

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            decoder_input_ids=None,
            decoder_attention_mask=None,
            head_mask=None,
            decoder_head_mask=None,
            cross_attn_head_mask=None,
            encoder_outputs=None,
            past_key_values=None,
            inputs_embeds=None,
            decoder_inputs_embeds=None,
            labels=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            neg_nl_input_ids=None,
            neg_nl_attention_mask=None
    ):
        return self.forward_search(input_ids=input_ids,
                                       attention_mask=attention_mask,
                                       decoder_input_ids=decoder_input_ids,
                                       decoder_attention_mask=decoder_attention_mask,
                                       head_mask=head_mask,
                                       decoder_head_mask=decoder_head_mask,
                                       cross_attn_head_mask=cross_attn_head_mask,
                                       encoder_outputs=encoder_outputs,
                                       past_key_values=past_key_values,
                                       inputs_embeds=inputs_embeds,
                                       decoder_inputs_embeds=decoder_inputs_embeds,
                                       labels=labels,
                                       use_cache=use_cache,
                                       output_attentions=output_attentions,
                                       output_hidden_states=output_hidden_states,
                                       return_dict=return_dict,
                                       neg_nl_input_ids=neg_nl_input_ids,
                                       neg_nl_attention_mask=neg_nl_attention_mask)

        
    def forward_representation(
            self,
            input_ids=None,
            attention_mask=None,
            decoder_input_ids=None,
            decoder_attention_mask=None,
            head_mask=None,
            decoder_head_mask=None,
            cross_attn_head_mask=None,
            encoder_outputs=None,
            past_key_values=None,
            inputs_embeds=None,
            decoder_inputs_embeds=None,
            labels=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None):

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            encoder_outputs=encoder_outputs,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = outputs[0]  # last hidden state

        eos_mask = input_ids.eq(self.config.eos_token_id)

        if len(torch.unique(eos_mask.sum(1))) > 1:
            raise ValueError("All examples must have the same number of <eos> tokens.")
        sentence_representation = hidden_states[eos_mask, :].view(hidden_states.size(0), -1,
                                                                  hidden_states.size(-1))[
                                  :, -1, :
                                  ]
        return sentence_representation, outputs

    def forward_search(
            self,
            input_ids=None,
            attention_mask=None,
            decoder_input_ids=None,
            decoder_attention_mask=None,
            head_mask=None,
            decoder_head_mask=None,
            cross_attn_head_mask=None,
            encoder_outputs=None,
            past_key_values=None,
            inputs_embeds=None,
            decoder_inputs_embeds=None,
            labels=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            neg_nl_input_ids=None,
            neg_nl_attention_mask=None
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        code_representation, code_outputs = self.forward_representation(input_ids=input_ids,
                                                                        attention_mask=attention_mask,
                                                                        # decoder_input_ids=None,
                                                                        # decoder_attention_mask=decoder_attention_mask,
                                                                        # head_mask=head_mask,
                                                                        # decoder_head_mask=decoder_head_mask,
                                                                        # cross_attn_head_mask=cross_attn_head_mask,
                                                                        # encoder_outputs=encoder_outputs,
                                                                        # past_key_values=past_key_values,
                                                                        # inputs_embeds=inputs_embeds,
                                                                        # decoder_inputs_embeds=None,
                                                                        # labels=None,
                                                                        use_cache=use_cache,
                                                                        # output_attentions=output_attentions,
                                                                        # output_hidden_states=output_hidden_states,
                                                                        return_dict=return_dict)
        nl_representation, nl_outputs = self.forward_representation(input_ids=decoder_input_ids,
                                                                    attention_mask=decoder_attention_mask,
                                                                    # decoder_input_ids=None,
                                                                    # decoder_attention_mask=None,
                                                                    # head_mask=head_mask,
                                                                    # decoder_head_mask=decoder_head_mask,
                                                                    # cross_attn_head_mask=cross_attn_head_mask,
                                                                    # encoder_outputs=encoder_outputs,
                                                                    # past_key_values=past_key_values,
                                                                    # inputs_embeds=inputs_embeds,
                                                                    # decoder_inputs_embeds=None,
                                                                    # labels=None,
                                                                    use_cache=use_cache,
                                                                    # output_attentions=output_attentions,
                                                                    # output_hidden_states=output_hidden_states,
                                                                    return_dict=return_dict)

        neg_nl_representation, neg_nl_outputs = self.forward_representation(input_ids=neg_nl_input_ids,
                                                                            attention_mask=neg_nl_attention_mask,
                                                                            use_cache=use_cache,
                                                                            return_dict=return_dict)

        pos_sim = f.cosine_similarity(code_representation, nl_representation)
        neg_sim = f.cosine_similarity(code_representation, neg_nl_representation)

        loss = (0.413 - pos_sim + neg_sim).clamp(min=1e-6).mean()
        return loss

        # if not return_dict:
        #     output = (code_representation,) + code_outputs[1:]
        #     return ((loss,) + output) if loss is not None else output
        #
        # return Seq2SeqSequenceClassifierOutput(
        #     loss=loss,
        #     logits=code_representation,
        #     past_key_values=code_outputs.past_key_values,
        #     decoder_hidden_states=code_outputs.decoder_hidden_states,
        #     decoder_attentions=code_outputs.decoder_attentions,
        #     cross_attentions=code_outputs.cross_attentions,
        #     encoder_last_hidden_state=code_outputs.encoder_last_hidden_state,
        #     encoder_hidden_states=code_outputs.encoder_hidden_states,
        #     encoder_attentions=code_outputs.encoder_attentions,
        # )

    def evaluate_search(self,
                        query_dataloader: torch.utils.data.dataloader.DataLoader,
                        codebase_dataloader: torch.utils.data.dataloader.DataLoader,
                        metrics_prefix: str):

        self.set_model_mode(enums.MODEL_MODE_CLS)
        self.eval()

        # embed query and codebase
        with torch.no_grad():
            logger.info('(1/3) Embedding search queries')
            query_vectors = []
            ref_urls = []
            for _, batch in enumerate(tqdm(query_dataloader)):
                urls = batch.pop('urls')
                ref_urls += urls
                batch = inputs_to_cuda(batch)
                representation, outputs = self.forward_representation(**batch)  # representation: [B, H]
                representation = representation.cpu().numpy()   # [B, H]
                query_vectors.append(representation)
            query_vectors = np.vstack(query_vectors)    # [len_query, H]

            logger.info('(2/3) Embedding candidate codes')
            code_vectors = []
            code_urls = []
            for _, batch in enumerate(tqdm(codebase_dataloader)):
                urls = batch.pop('urls')
                code_urls += urls
                batch = inputs_to_cuda(batch)
                representation, outputs = self.forward_representation(**batch)
                representation = representation.cpu().numpy()
                code_vectors.append(representation)
            code_vectors = np.vstack(code_vectors)  # [len_code, H]

            # calculate MRR
            logger.info('(3/3) Calculating metrics')
            scores = []
            ranks = []
            can_urls = []
            can_sims = []
            for query_vector, ref_url in tqdm(zip(query_vectors, ref_urls), total=len(query_vectors)):
                sims = []
                for code_vector, code_url in zip(code_vectors, code_urls):
                    sim = f.cosine_similarity(torch.from_numpy(code_vector).unsqueeze(0),
                                              torch.from_numpy(query_vector).unsqueeze(0))
                    sims.append((code_url, sim.item()))
                sims.sort(key=lambda item: item[1], reverse=True)

                sims = sims[:1000]
                can_urls.append(sims[0][0])
                can_sims.append(sims[0][1])

                rank = -1
                for index, (url, sim) in enumerate(sims):
                    if url == ref_url:
                        rank = index + 1
                ranks.append(rank)
                score = 1 / rank if rank != -1 else 0
                scores.append(score)

        self.train()
        self.set_model_mode(enums.MODEL_MODE_SEARCH)

        results = {f'{metrics_prefix}_mrr': np.mean(scores),
                   f'{metrics_prefix}_ranks': ranks,
                   f'{metrics_prefix}_ref_urls': ref_urls,
                   f'{metrics_prefix}_can_urls': can_urls,
                   f'{metrics_prefix}_can_sims': can_sims}
        return results


