# barcode-bias

Code and data accompanying McGee et al. _Improving the accuracy of bulk fitness assays
by correcting barcode processing biases._ bioRxiv (2023).

## Code

`bias_correction.py` contains a python module implementing our barcode bias correction method.

You can run this bias correction method using the command line or as part of your own python scripts, as seen in the examples below and in the [case study analysis notebook](https://github.com/ryansmcgee/barcode-bias/blob/main/case-study/step1_run-bias-correction.ipynb).

Further documentation about code usage is forthcoming. Questions regarding this code can be directed to ryansmcgee@gmail.com.

### Running the method from the command line

Run `bias_correction.py` as a script from the command line, specifying the paths of input files and the desired output directory as arguments. 

For example:
```
python bias_correction.py --counts ./case-study/data/kinsler-2020-preprocessed-data/counts.csv --samples ./case-study/data/kinsler-2020-preprocessed-data/samples.csv --variants ./case-study/data/kinsler-2020-preprocessed-data/variants_withNeutralGroups.csv --config ./case-study/analysis-config.json -outdir case-study/results/ 
```

### Interfacing with the method within python scripts

Our method is implemented using an object-oriented structure that makes it easy to interface with as a "black box" or in a more "hands on" manner as desired.

For example, executing the method within your own script can be as simple as:
```
# Set up bias correction object
debiaser = BiasCorrection(counts='./data/kinsler-2020-preprocessed-data/counts.csv', 
                          samples='./data/kinsler-2020-preprocessed-data/samples.csv', 
                          variants='./data/kinsler-2020-preprocessed-data/variants_withNeutralGroups.csv',
                          config='./analysis-config.json',
                          outdir='./results/')

# Perform bias correction
debiaser.run()
```

## Case Study Results

We applied our bias inference and correction method to the bulk fitness assay data from Kinsler et al. (2020) (refer to our preprint for more information). Input data, jupyter notebooks that replicate the execution of our method, results data, and figures can be found in the `case-study` directory.
