# DialoGen: Generalized Long-Range Context Representation for Dialogue Systems

1. Environment set up 
Create an environment with python 3.8 and install the required packages
```console
❱❱❱ pip install -r requirements.txt
```

2. Set up DailyDialog Data
	
 	a. Download DailyDialog dataset from the following link: http://yanran.li/files/ijcnlp_dailydialog.zip
	b. Unzip ijcnlp_dailydialog.zip
	c. Download multi-reference test data for DailyDialog from the following link: https://raw.githubusercontent.com/prakharguptaz/multirefeval/master/multiref-dataset/multireftest.json
	d. Copy the "multireftest.json" file to the "ijcnlp_dailydialog" directory
	e. Copy the modified "ijcnlp_dailydialog" directory to the root folder of the codebase  

3. Train Encoder (DailyDialog)
	a. python train_encoder.py -path=<enc_path> -src_file=train_encoder.py -model_file=encoder_model.py

4. Train Decoder (DailyDialog)
	a. DialoGen (with Z_t and top-k/last-k): python train_decoder.py -path=<dec_path> -src_file=train_decoder.py -model_file=decoder_model.py -gpt=large -enc_dir=<enc_path> -dec_type=2 -max_context=4 -keep_k=2 -bow

	b. DialoGen (only Z_t): python train_decoder.py -path=<dec_path> -src_file=train_decoder.py -model_file=decoder_model.py -gpt=large  -enc_dir=<enc_path> -dec_type=1 -bow

	c. DialoGen (only top-k/last-k): python train_decoder.py -path=<dec_path> -src_file=train_decoder.py -model_file=decoder_model.py -gpt=large -enc_dir=<enc_path> -dec_type=0 -max_context=4 -keep_k=2

6. Post-processing generated dialogues (DailyDialog)
	a. DailyDialog references have a particular text format. Run the following command to match the format approximately. 
	b. python post_process_dailydialog.py -in=<file_name>
	c. No need to run the post-processing script for DialoGen as the conversion are already applied during the output generation. This script is meant for to convert the output of other baselines.

7. Evaluation (DailyDialog)
	a. python compute_metrics.py -in=<result_path> -hyp=<hyp_file>
	b. Note: <result_path> is the directory that contains the <hyp_file> and the <ref_file>. By default, <result_path> = <dec_path>/result

8. Human Evaluation Data for DailyDialog
	a. Directory named "human_evaluation" contains all the soure data that has been used to conduct the human evaluation. Each file in the directory shows the context and the response generated using the four models: A) DialoGen, B) DialogVED, C) DialoFlow, and D) DialoGPT.

9. Train Encoder (MultiWOZ)
	a. Download MultiWOZ 2.1 dataset from https://github.com/budzianowski/multiwoz/tree/master/data followed by the required pre-processing instruction in https://github.com/budzianowski/multiwoz. 
	b. Create a directory named "multiwoz" and copy the following files into it - data.json, valListFile.txt, and testListFile.txt, train_dials.json, dev_dials.json, and test_dials.json.
	c. python train_encoder_multiwoz.py -path=<enc_path> -src_file=train_encoder_multiwoz.py -model_file=encoder_model_multiwoz.py

10. Experiment with MultiWOZ dataset
	a. Download or clone the SOM-DST (https://github.com/clovaai/som-dst) and Trippy (https://gitlab.cs.uni-duesseldorf.de/general/dsml/trippy-public/-/tree/master) code repository.
	b. Replace utils/data_utils.py in the SOM-DST codebase with the one provided by us. Set the path of the DialoGen encoder out directory (dialogen_encoder_output) correctly.
	c. Replace file dataset_multiwoz21.py in the Trippy codebase with the one provided by us. Set the path of the DialoGen encoder out directory (dialogen_encoder_output) correctly.
	d. Train SOM-DST and Trippy with the default parameters following the instruction given in the respective code repositories.
