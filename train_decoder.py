import os
import torch
import math
import random
import argparse
import shutil
import logging
import time
import json
import nltk
from nltk.tokenize import word_tokenize
import string
import re
from encoder_model import Encoder
from decoder_model import Decoder
from transformers import AdamW, get_linear_schedule_with_warmup
from datetime import datetime
from transformers import BertTokenizer, GPT2Tokenizer, BertModel

#-----------------------------------------
#Sample commnd to run the script 
#python train_decoder.py -path=<decoder_model_path> -src_file=train_decoder.py -model_file=decoder_model.py -gpt=large -enc_dir=<trained_encoder_model_path> -dec_type=2 -max_context=4 -keep_k=2 -bow
#-----------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument('-path','--path', help='path of the model directiory', required=True)
parser.add_argument('-src_file','--src_file', help='path of the source file', required=True)
parser.add_argument('-model_file','--model_file', help='path of the model class file', required=True)
parser.add_argument('-loss_type','--loss_type', help='Loss Type', default='l1', choices=['l1', 'rmse'])
parser.add_argument('-enc_dir','--enc_dir', help='Encoder directory', required=True)
parser.add_argument('-dec_type','--dec_type', help='Decoder Type', type=int, required=False, default=2)
parser.add_argument('-bow','--bow', help='Use BoW loss', default=False, action='store_true')
parser.add_argument('-rnn_type','--rnn_type', help='RNN Type (lostm/gru)', default='gru', choices=['gru', 'lstm'])
parser.add_argument('-n_layers','--n_layers', help='RNN layers', type=int, required=False, default=2)
parser.add_argument('-dropout','--dropout', help='dropout', type=float, required=False, default=0.2)
parser.add_argument('-bert','--bert', help='Model size of BERT', default='base', choices=['base', 'large'])
parser.add_argument('-gpt','--gpt', help='Model size of GPT', default='large', choices=['base', 'medium', 'large'])
parser.add_argument('-max_context','--max_context', help='Maximum context utterances', type=int, required=False, default=4)
parser.add_argument('-last_k','--last_k', help='Use last k context', default=False, action='store_true')
parser.add_argument('-keep_k','--keep_k', help='Keep last k', type=int, required=False, default=2)
parser.add_argument('-epochs','--epochs', help='training epochs', type=int, required=False, default=10)
parser.add_argument('-grad_acc','--grad_acc', help='gradient accuumulation steps', type=int, required=False, default=4)
parser.add_argument('-lr','--lr', help='Learning rate', type=float, required=False, default=1e-5)
parser.add_argument('-wr','--wr', help='Warm-up ratio', type=float, required=False, default=0.1)
parser.add_argument('-max_norm','--max_norm', help='Max norm', type=float, required=False, default=1.0)
parser.add_argument('-a1','--a1', help='Weight of encoder BoW loss', type=float, required=False, default=1.0)
parser.add_argument('-a2','--a2', help='Weight of encoder pred loss', type=float, required=False, default=1.0)
parser.add_argument('-a3','--a3', help='Weight of decoder BoW loss', type=float, required=False, default=0.5)
parser.add_argument('-skip_train','--skip_train', help='Skip Training', default=False, action='store_true')
parser.add_argument('-skip_gen','--skip_gen', help='Skip Generation', default=False, action='store_true')
parser.add_argument('-beam','--beam', help='beam size', type=int, required=False, default=5)

args = vars(parser.parse_args())
model_dir = args['path']
src_file = args['src_file']
model_src_file = args['model_file']
loss_type = args['loss_type']
encoder_dir = args['enc_dir']
dec_type = args['dec_type']
use_bow_loss = args['bow']
rnn_type = args['rnn_type']
n_layers = args['n_layers']
dropout = args['dropout']
bert_size = args['bert']
gpt_size = args['gpt']
max_context = args['max_context']
use_last_k = args['last_k']
keep_k = args['keep_k']
epochs =  args['epochs']
grad_acc =  args['grad_acc']
learning_rate = args['lr']
warmup_ratio = args['wr']
max_norm = args['max_norm']
a1 = args['a1']
a2 = args['a2']
a3 = args['a3']
skip_train = args['skip_train']
skip_gen = args['skip_gen']
beam_size = args['beam']

test_run = False
#Uncomment this code if you want to test the code using a small set of data
#test_run = True 

if(skip_train and skip_gen):
    print("You have skipped both training and generation")
    exit(0)
    
print(f"BERT size :: {bert_size}")
print(f"GPT size :: {gpt_size}")
print(f"dec_type :: {dec_type}")
print(f"Use BoW loss :: {use_bow_loss}")
print(f"Max context : {max_context}")
print(f"Use last-k :: {use_last_k}")
if(not use_last_k):
    print(f"keep_k :: {keep_k}")
print("Path of the model directory : {}".format(model_dir))
print("Path of the encoder model directory : {}".format(encoder_dir))

if(os.path.isdir(model_dir)):
    if(not skip_train):
        print("Model Directory exists.")
        exit(0)
else:
    os.mkdir(model_dir)
    print(f"Model directory {model_dir} created.")
    
BERT_MODEL = "bert-base-uncased"
GPT_MODEL = "microsoft/DialoGPT-small"    
input_size = 768
hidden_size = 768   
if(bert_size=="large"):
    BERT_MODEL = "bert-large-uncased"
    input_size = 1024
if(gpt_size=="medium"):
    GPT_MODEL = "microsoft/DialoGPT-medium"
    hidden_size = 1024
elif(gpt_size=="large"):
    GPT_MODEL = "microsoft/DialoGPT-large"
    hidden_size = 1280

SEED = 10
random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)      
    device = torch.device("cuda")
    print("Using GPU")
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

bert_tokenizer = BertTokenizer.from_pretrained(BERT_MODEL)
gpt_tokenizer = GPT2Tokenizer.from_pretrained(GPT_MODEL)
vocab_size = bert_tokenizer.vocab_size
gpt_vocab_size = gpt_tokenizer.vocab_size

if(not skip_train): 
    shutil.copy(src_file, model_dir)
    shutil.copy(model_src_file, model_dir)
    
    #Setting log file
    log_file = os.path.join(model_dir, 'log_decoder.txt')
    logging.basicConfig(filename=log_file, filemode='a', 
                        format='%(asctime)s %(message)s', 
                        datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
    logging.Formatter.converter = time.gmtime
    logger = logging.getLogger(__name__)
    
#-----------------------------------------


def emptyCudaCache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
def createFile(file_name):
    fp = open(file_name, "w", encoding="utf-8")
    return fp

def closeFile(fp_list):
    for fp in fp_list:
        fp.close()

#Retrieves the encoder output: X_t (context_enc), b_{t+1}' (context_pred), and relevance score \alpha (wt)        
def get_encoder_prediction(data, enc_model):
    id_list = list(range(len(data)))
    with torch.no_grad():
        for idx in id_list:
            utt_list = data[idx]
            loss_dict, context_enc_list, wt_list, context_pred_list = enc_model(utt_list, a1, a2, False, device)
            for turn in range(len(utt_list)):
                if(turn>0):
                    data[idx][turn]['context_enc'] = context_enc_list[turn-1].cpu()
                    data[idx][turn]['wt'] = wt_list[turn-1]
                    data[idx][turn]['context_pred'] = context_pred_list[turn-1].cpu()
    return data

#Loads and prepares dailydialog data for training/testing
def getData(mode, bert, enc_model):
    file_name = os.path.join("ijcnlp_dailydialog", mode, f"dialogues_{mode}.txt")
    lines = []
    with open(file_name, 'r') as f:
        lines = f.readlines()

    total_turns = 0
    dialog_data = []
    conv_num = 0
    for line in lines:
        conv_num+=1
            
        arr = line.strip().split('__eou__')
        utt_list = []
        flag = False
        for j in range(len(arr)-1):
            utt = {}
            utt_text = arr[j].strip()
            utt['txt'] = utt_text
            with torch.no_grad():
                bert_ids = bert_tokenizer.encode(utt_text.lower(), truncation=True, max_length=64, return_tensors='pt').to(device)
                utt['bert_enc'] = torch.mean(bert(input_ids=bert_ids)[2][-2], dim=1).squeeze().cpu()
                
                bow_ids = bert_tokenizer.encode(utt_text.lower(), truncation=True, max_length=64, return_tensors='pt').squeeze(0)
                bow_ids[0] = -100 #Mask <cls> token
                bow_ids[-1] = -100 #Mask <sep> token
                utt['bow_ids']  = bow_ids
                
                eos = gpt_tokenizer.encode(gpt_tokenizer.eos_token, return_tensors='pt')
                gpt_ids = gpt_tokenizer.encode(utt_text.lower(), truncation=True, max_length=64, return_tensors='pt')
                utt['gpt_ids'] = torch.cat((gpt_ids, eos), 1)
                #Use gpt token ids without <eos> token for bow loss
                utt['bow_ids_gpt'] = gpt_tokenizer.encode(utt_text.lower(), truncation=True, max_length=64, return_tensors='pt').squeeze(0)
                
            utt_list.append(utt) 
        total_turns += (len(utt_list)-1)
        dialog_data.append(utt_list)
        if(test_run and conv_num==20):
            break
        
    if(not skip_train):
        logger.info(f"{mode} data: #Conversations : {len(dialog_data)} : #Turns : {total_turns}")
        
    dialog_data = get_encoder_prediction(dialog_data, enc_model)
    return dialog_data, total_turns

#Returns evaluation loss of the given model on the given dataset
def evaluate_loss(data, model):
    total_count = 0
    total_loss = 0.0
    if(use_bow_loss):
        total_lm_loss = 0.0
        total_bow_loss = 0.0
    
    model.eval()
    id_list = list(range(len(data)))
    with torch.no_grad():
        for idx in id_list:
            utt_list = data[idx]  
            total_turns = len(utt_list)
            for turn in range(1, total_turns):
                loss, loss_dict = model(utt_list, turn, max_context, use_last_k, keep_k, a3, device)
                total_loss += loss_dict['loss']
                if(use_bow_loss):
                    total_lm_loss += loss_dict['lm']
                    total_bow_loss += loss_dict['bow']
            total_count+=(total_turns-1)
    
    loss_dict = {}
    loss_dict['loss'] = total_loss/total_count
    if(use_bow_loss):
        loss_dict['lm'] = total_lm_loss/total_count
        loss_dict['bow'] = total_bow_loss/total_count
    return loss_dict

#Returns total number of training instance in the next n (grad_acc_step) conversations
def get_batch_size(id_list, data, count, grad_acc_step):
    if(count>=len(id_list)):
        return 0
    else:
        bs = 0
        if((count+grad_acc_step-1)<len(id_list)):
            n = grad_acc_step
        else:
            n = len(id_list)-count
        for i in range(n):
            idx = id_list[count+i]
            bs += (len(data[idx])-1)
        return bs
            
#-----------------------------------------

#Load Encoder Model
enc_model_file = os.path.join(encoder_dir, f"encoder.pt")
enc_model = Encoder(input_size, rnn_type, n_layers, vocab_size, dropout, loss_type)
enc_model.load_state_dict(torch.load(enc_model_file))
enc_model.to(device)
enc_model.eval()
print("Encoder model loaded")

#Load Data
print("Loading Data ...")
bert = BertModel.from_pretrained(BERT_MODEL, output_hidden_states = True)
bert.to(device)
if(not skip_train):
    train_data, train_turns = getData("train", bert, enc_model)
    validation_data, validation_turns  = getData("validation", bert, enc_model)
test_data, test_turns = getData("test", bert, enc_model)
print("Data Loaded.")
bert = bert.cpu()
bert = None
enc_model = enc_model.cpu()
enc_model = None
emptyCudaCache()

#-----------------------------------------

#Logging details
if(not skip_train):
    logger.info(f"BERT Model : {BERT_MODEL}")
    logger.info(f"GPT Model : {GPT_MODEL}")
    logger.info(f"Max context : {max_context}")
    logger.info(f"Decoder type : {dec_type}")
    logger.info(f"Use BoW loss : {use_bow_loss}")
    logger.info(f"Use lask k : {use_last_k}")
    if(not use_last_k):
        logger.info(f"Keep last k : {keep_k}")
    logger.info(f"Input size : {input_size}")
    logger.info(f"Hidden size : {hidden_size}")
    logger.info(f"GPT Vocab size : {gpt_vocab_size}")
    logger.info(f"RNN type : {rnn_type}")
    logger.info(f"RNN layers : {n_layers}")
    logger.info(f"Dropout ratio : {dropout}")
    logger.info(f"Learning rate : {learning_rate}")
    logger.info(f"Warm-up ratio : {warmup_ratio}")
    logger.info(f"Max norm : {max_norm}")
    if(use_bow_loss):
        logger.info(f"Weight of Bow loss : {a3}")
    logger.info(f"Training epochs : {epochs}, Batch size : 1, Gradient accumulation step : {grad_acc}")

#-----------------------------------------

#Training Function
def train():
    #Load Model
    model = Decoder(GPT_MODEL, dec_type, use_bow_loss, input_size, hidden_size, gpt_vocab_size, dropout)
    model.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr = learning_rate)
    total_steps = (len(train_data)//grad_acc) * epochs
    warmup_steps = math.floor(total_steps*warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    logger.info(f"Linear schedule details:: total_steps : {total_steps}, warmup_steps : {warmup_steps}")
    
    best_loss = 999999.0
    best_epoch = -1
    model_file = os.path.join(model_dir, f"decoder.pt")
    logger.info("Training started...")
    logger.info("-"*40)

    for epoch in range(epochs):
        logger.info(f"EPOCH {epoch} started ...")
        emptyCudaCache()
        
        id_list = list(range(len(train_data)))
        random.shuffle(id_list)
        
        model.train()
        model.zero_grad(set_to_none=True)
        batch_count = 0 
        batch_size = get_batch_size(id_list, train_data, 0, grad_acc) #Effectve batch size for next n (grad_acc) conversations
        
        for count,idx in enumerate(id_list):
            utt_list = train_data[idx]    
            total_turns = len(utt_list)
            for turn in range(1, total_turns):
                loss, loss_dict = model(utt_list, turn, max_context, use_last_k, keep_k, a3, device)
                loss = loss/batch_size
                loss.backward()
            batch_count+=1
            
            if(batch_count%grad_acc == 0 or batch_count>=len(train_data)):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                optimizer.step()
                scheduler.step()
                model.zero_grad(set_to_none=True)
                batch_size = get_batch_size(id_list, train_data, count+1, grad_acc)
                
            if(batch_count%2000==0):
                logger.info(f"{batch_count} batches trained")   

        logger.info(f"Model trained for Epoch {epoch}")
        validation_loss = evaluate_loss(validation_data, model)
        test_loss = evaluate_loss(test_data, model)
        
        logger.info(f"Epoch {epoch} : Validation loss = {validation_loss} : Test loss = {test_loss}")
        logger.info(f"Epoch {epoch} completed.")
        
        if(validation_loss['loss']<best_loss):
            best_loss = validation_loss['loss']
            best_epoch = epoch
            if os.path.exists(model_file):
                os.remove(model_file)
            torch.save(model.state_dict(), model_file)
            logger.info(f"Epoch {epoch} model saved.") 
        else:
            if(epoch>(best_epoch+2)):
                logger.info(f"Early stopping at epoch {epoch}.") 
                break
        logger.info("-"*40)

    logger.info("-"*40)
    logger.info(f"Best Epoch : {best_epoch}") 
    logger.info("Loading best model...") 
    emptyCudaCache()
    model.load_state_dict(torch.load(model_file))
    model.to(device)

    logger.info("Computing train, validation and test loss of best model.") 
    train_loss = evaluate_loss(train_data, model)
    validation_loss = evaluate_loss(validation_data, model)
    test_loss = evaluate_loss(test_data, model)
    logger.info(f"Train loss = {train_loss}")
    logger.info(f"Validation loss = {validation_loss}")
    logger.info(f"Test loss = {test_loss}")
    logger.info("-"*40)
    logger.info("-"*40)
    return model

#Load multi-reference test data for dailydialog
def getReference(test_data):
    filename = "ijcnlp_dailydialog/multireftest.json"
    ref_data = []
    with open(filename, 'r') as f:
        for line in f:
            ref_data.append(json.loads(line))
    
    dialog_data = []
    total_turns = 0
    conv_num = 0
    for i,dt in enumerate(ref_data):
        conv_num+=1
        dialog = dt['dialogue']
        utt_list = []
        for j in range(len(dialog)):
            utt = {}
            utt_text = dialog[j]['text'].strip()
            utt['txt'] = utt_text
            if 'responses' in dialog[j]:
                utt['responses'] = dialog[j]['responses']

            utt['gpt_ids'] = test_data[i][j]['gpt_ids']
            if(j>0):
                utt['context_enc'] = test_data[i][j]['context_enc']
                utt['wt'] = test_data[i][j]['wt']
                utt['context_pred'] = test_data[i][j]['context_pred']
            utt_list.append(utt)
        total_turns += (len(utt_list)-1)
        dialog_data.append(utt_list)
        if(test_run and conv_num==2):
            break
        
    print(f"Test data: #Conversations : {len(dialog_data)} : #Turns : {total_turns}")
    return dialog_data

#Generate decoder result
def generate_response(test_data, model, model_dir):
    data = getReference(test_data)
    result_dir = os.path.join(model_dir, "result")
    if not os.path.isdir(result_dir):
        os.mkdir(result_dir)
        
    fp_list = []
    for i in range(5):
        fn = os.path.join(result_dir, f"ref{i}.txt")
        fp = createFile(fn)
        fp_list.append(fp)

    hyp_file = os.path.join(result_dir, f"hyp_{beam_size}.txt")
    fp = createFile(hyp_file)
    fp_list.append(fp)
    
    conv_len_file = os.path.join(result_dir, f"conv_length.txt")
    fp = createFile(conv_len_file)
    fp_list.append(fp)
    
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print(f"Generation started : {current_time}")
    
    result_gen = {}
    model.eval()
    id_list = list(range(len(data)))
    with torch.no_grad():
        for idx in id_list:
            res_gen = []
            utt_list = data[idx]
            total_turns = len(utt_list)
            for turn in range(total_turns):
                if(turn<(total_turns-1)):
                    turn_gen = {}
                    turn_gen['1.turn'] = turn
                    turn_gen['2.txt'] = utt_list[turn]['txt']
                    turn_gen['3.res'] = utt_list[turn]['responses']
                    res_gen.append(turn_gen)
                
                if(turn>0):
                    utt_pred1 = model.decodeBeam(utt_list, turn, max_context, use_last_k, keep_k, beam_size, gpt_tokenizer, device)
                    #Post-processing of generated dialogue
                    pred_tokens = word_tokenize(utt_pred1.lower())
                    utt_pred = ' '.join(pred_tokens) + "\n"
                    utt_pred = utt_pred.replace("’","'")
                    utt_pred = re.sub("( ')(\s+)*([a-z]+)", r'\1\3', utt_pred)
                    utt_pred = re.sub("([a-z]+)(n| n| n |n )(')(t| t)", r"\1 n't", utt_pred)
                    utt_pred = re.sub("(\s+)([a-z]+)(\.| \.|\. )([a-z]+)(\s+)", r"\1\2 . \4\5", utt_pred)
                    
                    res_gen[turn-1]['4.gen'] = utt_pred.strip()
                    if(not use_last_k):
                        res_gen[turn-1]['5.att'] = utt_list[turn]['wt']
                    
                    #Write references
                    for j in range(5):
                        fp_list[j].write(utt_list[turn-1]['responses'][j]+"\n")
                    fp_list[5].write(utt_pred)
                    fp_list[6].write(f"{total_turns}\n")
                
            result_gen[idx] = res_gen
            
            if((idx+1)%100==0):
                now = datetime.now()
                current_time = now.strftime("%H:%M:%S")
                print(f"{idx} conversations completed : {current_time}")
               
    closeFile(fp_list)   
    filename = os.path.join(model_dir, 'result_gen.json')
    result_file = open(filename, "w")
    result_file.write(json.dumps(result_gen, indent=4, sort_keys=True))
    result_file.close()
    
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print(f"Generation completed : {current_time}")
    
#-----------------------------------------    

if(not skip_train):
    print("Training Started ...")
    model = train()
    print("Training Complete.")
    
if(not skip_gen):
    train_data = None
    valid_data = None
    emptyCudaCache()
    model_file = os.path.join(model_dir, f"decoder.pt")
    if(not os.path.exists(model_file)):
        print(f"Model file ({model_file}) does not exist.")
        exit(0)
    else:
        if(skip_train):
            model = Decoder(GPT_MODEL, dec_type, use_bow_loss, input_size, hidden_size, gpt_vocab_size, dropout)
            model.load_state_dict(torch.load(model_file))
            model.to(device)
            print("Model Loaded.")
            
        print("Generation Started ...")
        generate_response(test_data, model, model_dir)
        print("Generation Complete.")
        
print("done")
#-----------------------------------------