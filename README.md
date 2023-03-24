# DBDEMO Live demo

This repository contains the source for the live dbdemos demo, build for mobile phone usage during live events. 


## Compilation
Install boostrap: `npm i bootstrap`

Install sass: `npm install node-sass`

saas compilation: `node-sass docs/custom.scss -o docs/css --watch`

### Note on the Demo Data

Please do not submit personal information using this demo (use the generated pseudonyms). 

Data is purged after demos and the ML model will only keep pseudonyms having a firstname only.

Only the pseudonym and your item choice are submitted to the demo.


### Dataset used
The List of firstname used by the ML model is from https://www.data.gouv.fr/fr/datasets/liste-de-prenoms-et-patronymes/ - under Open Licence version 2.0