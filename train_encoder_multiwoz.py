import os
import torch
import math
import random
import argparse
import shutil
import logging
import time
import json
from encoder_model_multiwoz import Encoder
from transformers import AdamW, get_linear_schedule_with_warmup
from datetime import datetime
from transformers import BertTokenizer, GPT2Tokenizer, BertModel

#-----------------------------------------
#Sample command to run the code
#python train_encoder_multiwoz.py -path=$wd -src_file=train_encoder_multiwoz.py -model_file=encoder_model_multiwoz.py
#-----------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument('-path','--path', help='path of the model directiory', required=True)
parser.add_argument('-src_file','--src_file', help='path of the source file', required=True)
parser.add_argument('-model_file','--model_file', help='path of the model class file', required=True)
parser.add_argument('-rnn_type','--rnn_type', help='RNN Type (lstm/gru)', default='gru', choices=['gru', 'lstm'])
parser.add_argument('-n_layers','--n_layers', help='RNN layers', type=int, required=False, default=2)
parser.add_argument('-dropout','--dropout', help='dropout', type=float, required=False, default=0.2)
parser.add_argument('-bert','--bert', help='Model size of BERT', default='base', choices=['base', 'large'])
parser.add_argument('-epochs','--epochs', help='training epochs', type=int, required=False, default=30)
parser.add_argument('-grad_acc','--grad_acc', help='gradient accuumulation steps', type=int, required=False, default=4)
parser.add_argument('-lr','--lr', help='Learning rate', type=float, required=False, default=0.0005)
parser.add_argument('-wr','--wr', help='Warm-up ratio', type=float, required=False, default=0.1)
parser.add_argument('-max_norm','--max_norm', help='Max norm', type=float, required=False, default=1.0)
parser.add_argument('-loss_type','--loss_type', help='Loss Type', default='l1', choices=['l1', 'rmse', 'cos'])
parser.add_argument('-a1','--a1', help='Weight of BoW loss', type=float, required=False, default=1.0)
parser.add_argument('-a2','--a2', help='Weight of pred loss', type=float, required=False, default=1.0)
parser.add_argument('-skip_train','--skip_train', help='Skip Training', default=False, action='store_true')
parser.add_argument('-skip_gen','--skip_gen', help='Skip Generation', default=False, action='store_true')

args = vars(parser.parse_args())
model_dir = args['path']
src_file = args['src_file']
model_src_file = args['model_file']
rnn_type = args['rnn_type']
n_layers = args['n_layers']
dropout = args['dropout']
bert_size = args['bert']
epochs =  args['epochs']
grad_acc =  args['grad_acc']
learning_rate = args['lr']
warmup_ratio = args['wr']
max_norm = args['max_norm']
loss_type = args['loss_type']
a1 = args['a1']
a2 = args['a2']
skip_train = args['skip_train']
skip_gen = args['skip_gen']

test_run = False
#Uncomment this code if you want to test the code using a small set of data
#test_run = True 

if(skip_train and skip_gen):
    print("You have skipped both training and generation")
    exit(0)
    
print(f"BERT size : {bert_size}")
print(f"Loss Type : {loss_type}")
print("Path of the model directory : {}".format(model_dir))

if(os.path.isdir(model_dir)):
    if(not skip_train):
        print("Model Directory exists.")
        exit(0)
else:
    os.mkdir(model_dir)
    print(f"Model directory {model_dir} created.")
    
BERT_MODEL = "bert-base-uncased"   
input_size = 768  
if(bert_size=="large"):
    BERT_MODEL = "bert-large-uncased"
    input_size = 1024

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
bert = BertModel.from_pretrained(BERT_MODEL, output_hidden_states = True)
bert.to(device)
vocab_size = bert_tokenizer.vocab_size

if(not skip_train): 
    shutil.copy(src_file, model_dir)
    shutil.copy(model_src_file, model_dir)
    
    #Setting log file
    log_file = os.path.join(model_dir, 'log_encoder_multiwoz.txt')
    logging.basicConfig(filename=log_file, filemode='a', 
                        format='%(asctime)s %(message)s', 
                        datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
    logging.Formatter.converter = time.gmtime
    logger = logging.getLogger(__name__)
    
#-----------------------------------------

def emptyCudaCache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
def loadJson(data_file):
    if os.path.isfile(data_file):
        with open(data_file, 'r') as read_file:
            data = json.load(read_file)
            return data
        
def load_list_file(list_file):
    with open(list_file, 'r') as read_file:
        dialog_id_list = read_file.readlines()
        dialog_id_list = [l.strip('\n') for l in dialog_id_list]
        return dialog_id_list
    return

def get_tokens(utt_text):
    utt = {}
    with torch.no_grad():
        bert_ids = bert_tokenizer.encode(utt_text.lower(), truncation=True, max_length=64, return_tensors='pt').to(device)
        utt['txt'] = utt_text
        utt['bert_enc'] = torch.mean(bert(input_ids=bert_ids)[2][-2], dim=1).squeeze().cpu()
        bow_ids = bert_tokenizer.encode(utt_text.lower(), truncation=True, max_length=64, return_tensors='pt').squeeze(0).cpu()
        bow_ids[0] = -100 #Mask <cls> token
        bow_ids[-1] = -100 #Mask <sep> token
        utt['bow_ids'] = bow_ids
    return utt

#Loads and prepares dailydialog data for training/testing
def getData(mode, raw_data):
    #Set the path of multiwoz data that contains the [train|dev|test]_dials.json files
    multiwoz_path = "multiwoz"
    dials_path = os.path.join(multiwoz_path, mode+"_dials.json")
    dials = loadJson(dials_path)
    dials_data = {}
    for i,d in enumerate(dials):
        dials_data[d['dialogue_idx']] = d
    
    conv_num = 0
    total_turns = 0
    dialog_data = {}
    for k,d in raw_data:
        if k in dials_data:
            conv_num+=1
            log = dials_data[k]
            dialog = log['dialogue']
            utt_list = []
            for i in range(len(dialog)):
                sys = dialog[i]['system_transcript']
                usr = dialog[i]['transcript']
                if(i>0):
                    utt_list.append(get_tokens(sys))
                utt_list.append(get_tokens(usr))
            if(len(d['log'])> len(utt_list)):
                v = d['log'][len(utt_list)]['text']
                utt_list.append(get_tokens(v))

            total_turns += int(len(utt_list)/2)
            dialog_data[k] = utt_list
            if(test_run and conv_num==20):
                break
        
    if(not skip_train):
        logger.info(f"{mode} data: #Conversations : {len(dialog_data)} : #Turns : {total_turns}")
    return dialog_data, total_turns

#Returns evaluation loss of the given model on the given dataset
def evaluate_loss(data, model):
    total_loss = 0.0
    total_bow_loss = 0.0   
    total_pred_loss = 0.0
    total_count = 0
    
    model.eval()
    #id_list = list(range(len(data)))
    id_list = [k for k in data]
    with torch.no_grad():
        for idx in id_list:
            utt_list = data[idx]          
            loss_dict, context_enc_list, wt_list, context_pred_list = model(utt_list, a1, a2, False, device)
            total_loss += loss_dict['loss']
            total_bow_loss += loss_dict['bow']
            total_pred_loss += loss_dict['next_utt']
            total_count+=len(wt_list)
                
    loss_dict = {}
    loss_dict['loss'] = total_loss/total_count
    loss_dict['bow'] = total_bow_loss/total_count
    loss_dict['next_utt'] = total_pred_loss/total_count
    return loss_dict

#Generate encoder result
def generate_result(mode, data, model, model_dir):
    result_gen = {}
    model.eval()
    id_list = [k for k in data]
    
    with torch.no_grad():
        for idx in id_list:
            res_gen = []
            utt_list = data[idx]
            loss_dict, context_enc_list, wt_list, context_pred_list = model(utt_list, a1, a2, False, device)
            for turn in range(len(utt_list)):
                if(turn%2==0):
                    continue
                turn_gen = {}
                turn_gen['1.usr'] = utt_list[turn-1]['txt']
                turn_gen['2.sys'] = utt_list[turn]['txt']
                if(turn>0):
                    p = int((turn-1)/2)
                    turn_gen['3.wt'] = wt_list[p]
                res_gen.append(turn_gen)
            result_gen[idx] = res_gen
       
    filename = os.path.join(model_dir, f"result_encoder_{mode}.json")
    result_file = open(filename, "w")
    result_file.write(json.dumps(result_gen, indent=4, sort_keys=True))
    result_file.close()
            
#-----------------------------------------

#Load Data
print("Loading Data ...")
#Load raw data

#Set the path of multiwoz data that contains the data.json, valListFile.txt, and testListFile.txt files
path = "multiwoz"
dialog_data_file = os.path.join(path, 'data.json')
dialog_data = loadJson(dialog_data_file)
dialog_id_list = list(set(dialog_data.keys()))
valid_list_file = os.path.join(path, 'valListFile.txt')
test_list_file = os.path.join(path, 'testListFile.txt')
valid_id_list = list(set(load_list_file(valid_list_file)))
test_id_list = load_list_file(test_list_file)
train_id_list = [did for did in dialog_id_list if did not in (valid_id_list + test_id_list)]
print('# of train dialogs:', len(train_id_list))
print('# of valid dialogs:', len(valid_id_list))
print('# of test dialogs :', len(test_id_list))
train_raw = [(k,v) for k, v in dialog_data.items() if k in train_id_list]
valid_raw = [(k,v) for k, v in dialog_data.items() if k in valid_id_list]
test_raw = [(k,v) for k, v in dialog_data.items() if k in test_id_list]
assert(len(train_raw) == len(train_id_list))
assert(len(valid_raw) == len(valid_id_list))
assert(len(test_raw) == len(test_id_list))

#if(not skip_train):
train_data, train_turns = getData("train", train_raw)
validation_data, validation_turns  = getData("dev", valid_raw)
test_data, test_turns = getData("test", test_raw)
bert = bert.cpu()
bert = None
emptyCudaCache()
print("Data Loaded.")

#-----------------------------------------

#Logging details
if(not skip_train):
    logger.info(f"BERT Model : {BERT_MODEL}")
    logger.info(f"Input size : {input_size}")
    logger.info(f"Vocab size : {vocab_size}")
    logger.info(f"RNN type : {rnn_type}")
    logger.info(f"RNN layers : {n_layers}")
    logger.info(f"Dropout ratio : {dropout}")
    logger.info(f"Learning rate : {learning_rate}")
    logger.info(f"Warm-up ratio : {warmup_ratio}")
    logger.info(f"Max norm : {max_norm}")
    logger.info(f"Loss Type : {loss_type}")
    logger.info(f"Weight of Bow loss : {a1}")
    logger.info(f"Weight of pred loss : {a2}")
    logger.info(f"Training epochs : {epochs}, Batch size : 1, Gradient accumulation step : {grad_acc}")

#-----------------------------------------

#Training Function
def train():
    model = Encoder(input_size, rnn_type, n_layers, vocab_size, dropout, loss_type)
    model.to(device)
    optimizer = AdamW(model.parameters(), lr = learning_rate, eps = 1e-8)
    total_steps = (len(train_data)//grad_acc) * epochs
    warmup_steps = 0
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    logger.info(f"Linear schedule details:: total_steps : {total_steps}, warmup_steps : {warmup_steps}")
    
    best_loss = 999999.0
    best_epoch = -1
    model_file = os.path.join(model_dir, f"encoder.pt")
    logger.info("Training started...")
    logger.info("-"*40)

    for epoch in range(epochs):
        logger.info(f"EPOCH {epoch} started ...")
        emptyCudaCache()
        
        #id_list = list(range(len(train_data)))
        id_list = [k for k in train_data]
        random.shuffle(id_list)
        
        model.train()
        model.zero_grad(set_to_none=True)
        batch_count = 0
        
        for count,idx in enumerate(id_list):
            utt_list = train_data[idx]
            loss = model(utt_list, a1, a2, True, device)
            loss = loss/grad_acc
            loss.backward()
            batch_count+=1
            
            if(batch_count%grad_acc == 0 or batch_count>=len(train_data)):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                optimizer.step()
                scheduler.step()
                model.zero_grad(set_to_none=True) 

        logger.info(f"Model trained for Epoch {epoch}")
        emptyCudaCache()
        validation_loss = evaluate_loss(validation_data, model)
        emptyCudaCache()
        test_loss = evaluate_loss(test_data, model)
        emptyCudaCache()
        
        logger.info(f"Epoch {epoch} : Validation loss = {validation_loss} : Test loss = {test_loss}")
        logger.info(f"Epoch {epoch} completed.")
        
        if(validation_loss['loss']<best_loss):
            best_loss = validation_loss['loss']
            best_epoch = epoch
            if os.path.exists(model_file):
                os.remove(model_file)
            torch.save(model.state_dict(), model_file)
            logger.info(f"Epoch {epoch} model saved.") 
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

#-----------------------------------------    

if(not skip_train):
    print("Training Started ...")
    model = train()
    print("Training Complete.")
    
if(not skip_gen):
    emptyCudaCache()
    model_file = os.path.join(model_dir, f"encoder.pt")
    if(not os.path.exists(model_file)):
        print(f"Model file ({model_file}) does not exist.")
        exit(0)
    else:
        if(skip_train):
            model = Encoder(input_size, rnn_type, n_layers, vocab_size, dropout, loss_type)
            model.load_state_dict(torch.load(model_file))
            model.to(device)
            print("Model Loaded.")
            
        print("Generation Started ...")
        generate_result("train", train_data, model, model_dir)
        generate_result("dev", validation_data, model, model_dir)
        generate_result("test", test_data, model, model_dir)
        print("Generation Complete.")
        
print("done")
#-----------------------------------------