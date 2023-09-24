import torch
import torch.nn as nn
import math
from transformers import GPT2LMHeadModel
import copy
import numpy as np
        
class Decoder(nn.Module):
    def __init__(self, model_name, dec_type, use_bow_loss, input_size, hidden_size, vocab_size, dropout):
        super().__init__()
        self.decoder = GPT2LMHeadModel.from_pretrained(model_name, output_hidden_states=True, output_attentions=True)
        self.dec_type = dec_type #0: only last-k/top-k, 1: only Z_t, 2: both Z_t and last-k/top-k 
        self.use_bow_loss = use_bow_loss
        
        if(dec_type>0):
            self.dropout = nn.Dropout(dropout)
            self.gelu = nn.GELU()
            self.context_enc_fc = nn.Linear(input_size*2, hidden_size) 
            self.context_enc_ln = nn.LayerNorm(hidden_size)
        
        if(use_bow_loss):
            self.bow_head = nn.Linear(hidden_size, vocab_size, bias=False)
            self.loss_bow = nn.CrossEntropyLoss(ignore_index=-100)
    
    def getLastK(self, turn, utt_list, max_context):
        k = max_context if(turn-max_context>=0) else turn 
        context_ids = utt_list[turn-k]['gpt_ids']
        for i in range(k-1):
            context_ids = torch.cat((context_ids, utt_list[turn-k+i+1]['gpt_ids']), dim=1)
        return context_ids
    
    def getContextIds(self, turn, utt_list, max_context, wt, keep_k):
        k = max_context if(turn-max_context>=0) else turn
        if(keep_k>0):
            context_ids1 = self.getLastK(turn, utt_list, keep_k)
        if(k<=keep_k and keep_k>0):
            return context_ids1
        else:
            wt_list = []
            for i,x in enumerate(wt):
                if(i<(turn-keep_k)):
                    wt_list.append((i, x))
            #Compute top (max_context - keep_k) utterances fro the remaining past history
            wt_list.sort(key = lambda x: x[1], reverse=True)
            wt_list2= [x[0] for x in wt_list[0:k-keep_k]]
            wt_list2.sort()
            context_ids = utt_list[wt_list2[0]]['gpt_ids']
            for i in range(k-keep_k-1):
                context_ids = torch.cat((context_ids, utt_list[wt_list2[i+1]]['gpt_ids']), dim=1)
            if(keep_k>0):
                context_ids = torch.cat((context_ids, context_ids1), dim=1)
            return context_ids
    
    def forward(self, utt_list, turn, max_context, use_last_k, keep_k, a3, device): 
        utt_ids = utt_list[turn]['gpt_ids'].to(device)
        wt = utt_list[turn]['wt']
        
        if(self.dec_type>0):
            context_enc = utt_list[turn]['context_enc'].to(device) #X_t
            context_pred = utt_list[turn]['context_pred'].to(device) #b_{t+1}'
            context_enc_final = torch.cat((context_enc, context_pred), -1) #Concatenating past and predicted context
            context_enc_final = self.context_enc_ln(self.context_enc_fc(self.dropout(self.gelu(context_enc_final))))
        
        #LM loss
        if(self.dec_type!=1):
            #Construct context for decoder using last-k/top-k strategy
            if(use_last_k):
                context_ids = self.getLastK(turn, utt_list, max_context).to(device) #last-k
            else:
                context_ids = self.getContextIds(turn, utt_list, max_context, wt, keep_k).to(device) #top-k
            input_ids = torch.cat((context_ids, utt_ids),dim=1).to(device)
            
        if(self.dec_type==0):
            #Use only last-k/top-k as context
            mask_tensor = -100*torch.ones(1,context_ids.size(1), dtype=context_ids.dtype).to(device)
            label = torch.cat((mask_tensor,utt_ids),dim=1).to(device)
            outputs = self.decoder(input_ids=input_ids, labels=label)
        elif(self.dec_type==1):
            #Use only Z_t as context
            mask_tensor = -100*torch.ones(1,1, dtype=utt_ids.dtype).to(device)
            label = torch.cat((mask_tensor,utt_ids),dim=1).to(device)
            inputs_embeds = torch.cat((context_enc_final.unsqueeze(0).unsqueeze(0), 
                                       self.decoder.transformer.wte(utt_ids)), dim=1)
            outputs = self.decoder(inputs_embeds=inputs_embeds, labels=label)
        else:
            #Use both Z_t and last-k/top-k as context
            mask_tensor = -100*torch.ones(1,context_ids.size(1)+1, dtype=context_ids.dtype).to(device)
            label = torch.cat((mask_tensor,utt_ids),dim=1).to(device)
            inputs_embeds = torch.cat((context_enc_final.unsqueeze(0).unsqueeze(0), 
                                       self.decoder.transformer.wte(input_ids)), dim=1)
            outputs = self.decoder(inputs_embeds=inputs_embeds, labels=label)
        lm_loss = outputs.loss
        
        #BoW loss
        if(self.use_bow_loss):
            bow_ids = utt_list[turn]['bow_ids_gpt'].to(device) 
            bow_logits = self.bow_head(context_enc_final)
            bow_logits = bow_logits.expand(bow_ids.size(0), -1)
            bow_loss = self.loss_bow(bow_logits, bow_ids)
        
        loss = (lm_loss + a3*bow_loss) if(self.use_bow_loss) else lm_loss
        loss_dict = {}
        with torch.no_grad():
            loss_dict['loss'] = loss.item()
            if(self.use_bow_loss):
                loss_dict['lm'] = lm_loss.item()
                loss_dict['bow'] = bow_loss.item()
            
        return loss, loss_dict
    
    #Beam search
    #The function is heavily inspired from https://github.com/ictnlp/DialoFlow/blob/main/generate.py
    def decodeBeam(self, utt_list, turn, max_context, use_last_k, keep_k, beam_size, gpt_tokenizer, device):
        max_length = 40
        min_length = 11
        penalty = 0.1
        eos = gpt_tokenizer.eos_token_id
        special_tokens_ids = [eos]
        
        utt_pred = ""
        with torch.no_grad():
            wt = utt_list[turn]['wt']
            
            if(self.dec_type>0):
                context_enc = utt_list[turn]['context_enc'].to(device) #X_t
                context_pred = utt_list[turn]['context_pred'].to(device) #b_{t+1}'
                context_enc_final = torch.cat((context_enc, context_pred), -1) #Concatenating past and predicted context
                context_enc_final = self.context_enc_ln(self.context_enc_fc(self.dropout(self.gelu(context_enc_final))))
                
            if(self.dec_type!=1):
                #Construct context for decoder using last-k/top-k strategy
                if(use_last_k):
                    context_ids = self.getLastK(turn, utt_list, max_context).to(device) #last-k
                else:
                    context_ids = self.getContextIds(turn, utt_list, max_context, wt, keep_k).to(device) #top-k
            
            hyplist = []
            if(self.dec_type==1):
                hyplist.append(([], 0., None))
            else:
                input_ids = context_ids
                hyplist.append(([], 0., copy.deepcopy(input_ids)))
            best_state = None
            comp_hyplist = []

            for i in range(max_length):
                new_hyplist = []
                argmin = 0
                for out, lp, st in hyplist:
                    if(self.dec_type==0):
                        outputs = self.decoder(input_ids=st.to(device))
                    elif(self.dec_type==1):
                        if(i==0):
                            inputs_embeds = context_enc_final.unsqueeze(0).unsqueeze(0).to(device)
                        else:
                            inputs_embeds = torch.cat((context_enc_final.unsqueeze(0).unsqueeze(0), 
                                           self.decoder.transformer.wte(st)), dim=1).to(device)
                        outputs = self.decoder(inputs_embeds=inputs_embeds) 
                    else:
                        inputs_embeds = torch.cat((context_enc_final.unsqueeze(0).unsqueeze(0), 
                                       self.decoder.transformer.wte(st)), dim=1).to(device)
                        outputs = self.decoder(inputs_embeds=inputs_embeds) 
                    
                    logits = outputs.logits
                    logp = nn.functional.log_softmax(logits, dim=-1)[:, -1, :]
                    lp_vec = logp.cpu().data.numpy() + lp
                    lp_vec = np.squeeze(lp_vec)
                    
                    if i< min_length:
                        lp_vec[eos] = -float("inf")
                    
                    if i >= min_length:
                        new_lp = lp_vec[eos] / (len(out) + 1)**penalty
                        comp_hyplist.append((out, new_lp))
                        if best_state is None or best_state < new_lp:
                            best_state = new_lp

                    count = 1
                    for o in np.argsort(lp_vec)[::-1]:
                        if o in special_tokens_ids:
                            continue
                        new_lp = lp_vec[o]
                        if len(new_hyplist) == beam_size:
                            if new_hyplist[argmin][1] < new_lp:
                                if(self.dec_type==1 and i==0):
                                    new_st = torch.tensor([[o]], dtype=utt_list[turn]['gpt_ids'].dtype).to(device)
                                else:
                                    new_st = copy.deepcopy(st)
                                    new_st = torch.cat((new_st, torch.tensor([[o]], dtype=new_st.dtype).to(new_st.get_device())), dim=1)
                                new_hyplist[argmin] = (out + [o], new_lp, new_st)
                                argmin = min(enumerate(new_hyplist), key=lambda h: h[1][1])[0]
                            else:
                                break
                        else:
                            if(self.dec_type==1 and i==0):
                                new_st = torch.tensor([[o]], dtype=utt_list[turn]['gpt_ids'].dtype).to(device)
                            else:
                                new_st = copy.deepcopy(st)
                                new_st = torch.cat((new_st, torch.tensor([[o]], dtype=new_st.dtype).to(new_st.get_device())), dim=1)
                            new_hyplist.append((out + [o], new_lp, new_st))
                            if len(new_hyplist) == beam_size:
                                argmin = min(enumerate(new_hyplist), key=lambda h: h[1][1])[0]
                        count += 1
                hyplist = new_hyplist
            if len(comp_hyplist) > 0:
                maxhyps = sorted(comp_hyplist, key=lambda h: -h[1])[:1]
                utt_pred = gpt_tokenizer.decode(maxhyps[0][0], skip_special_tokens=True).replace("\n", "") + "\n"
                
        return utt_pred