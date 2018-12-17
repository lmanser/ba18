# BA18 classification system
age classification program

# file structure
this file structure is necessary in order to run the program properly. If using the AgeingDatabaseReleaseII (not necessary if models are given), the structure is used "as-is". Meaning the structure when unzipping the downladed .zip file of the database doesn't have to be changed after unzipping.

AgeingDatabaseReleaseII/
	+ TCDSA_additional/
		+ TCDSA_ex/
			+ data/
			+ doc.csv
		+ UF_VAD_ex/
			+ data/
			+ doc.csv
	+ TCDSA_docs/
		+ doc_main.csv
	+ TCDSA_main/
appdata/
code/
	+ base.py
	+ Classes.py
	+ demo.py
	+ paths.py
 	+ sets.py
 	+ test.py
 	+ train.py
	+ featureExtraction.praat
	
# demo
In order to run the program for demonstration, change your working directory to ~/code/ and type:

	$ python3 demo.py
	
# full usage
In order to be able to run the whole system, the TCSDA data at the described location (see above) is necessary. Once the data is at the correct location, the order of operations is the following:

1) Feature segmentation and extraction (warning: takes about ~10 hours with the TCSDA data):

	$ python3 base.py

2) Computation of classifier models:

	$ python3 train.py

3) Evaluation of the trained models:

	$ python3 test.py