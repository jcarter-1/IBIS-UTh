## Install 
pip install ibis-uth

## Dependencies
Dependencies are written in both requirements.txt and pyproject.poml. <br>
pip install should import all neccessary dependencies however python verion >= 3.10 is needed

## Import

import ibis <br>
from ibis.IBIS_Main import IBIS

# IBIS
-----

IBIS is a probabilisitc Bayesian model to infer unique sample-specific initial thorium compositions estaimtes thoughtout a speleothem and "correct" apparent U—Th ages accordingly. 

## Why does this matter?
-----
* Ultimately the precision and accuracy of the U—Th method is dependent on this correct.
* Correction can bias ages dramatically, in particular for "dirtier" samples higher thorium content - low measured $^{230}$Th/$^{232}$Th activity ratios and also young samples with a lower "true" $^{230}$Th radiogenic budget.


## Age Equation


$$  \Delta_{\lambda} = \lambda_{230} -  \lambda_{234}$$


$$  \bigg[\bigg(\frac{^{230}Th}{^{238}U}\bigg)_A - \bigg(\frac{^{232}Th}{^{238}U}\bigg)_A \bigg(\frac{^{230}Th}{^{232}Th}\bigg)_{A0} e^{-\lambda_{230}t}\bigg]= 1 - e^{-\lambda_{230}t} + \bigg[ \bigg( \frac{^{234}U}{^{238}U}\bigg)_{A} - 1\bigg]\frac{\lambda_{230}}{\Delta_{\lambda}}(1 - e^{-\Delta_{\lambda}t})$$


* $\lambda_{230}$ - $^{230}$Th decay constant
* $\lambda_{234}$ - $^{234}$U decay constant
* $\bigg(\frac{^{230}Th}{^{238}U}\bigg)_A$ - measured activity ratio of $^{230}$Th to $^{238}$U
*  $\bigg(\frac{^{232}Th}{^{238}U}\bigg)_A$ - measured activity ratio of $^{232}$Th to $^{238}$U
*  $\bigg(\frac{^{234}U}{^{238}U}\bigg)_A$ - measured activity ratio of $^{234}$U to $^{238}$U
*  $\bigg(\frac{^{230}Th}{^{232}Th}\bigg)_{A0}$ - estiamte initial activity ratio at the time of formation/deposition/crystallization


A U-Th age requires the initial thorium $\bigg(\frac{^{230}Th}{^{232}Th}\bigg)_{A0}$ correction. In the age equation you need to guess a value for this component at the time of deposition and then it is brought forward in time to the present day. This present day initial is then used to correct the present day measured activity ratios. 


##### References
Kinsley, C.W., Carter, J.N. and Sharp, W.D., 2025. IBIS: an Integrated Bayesian approach for unique Initial thorium corrections and age-depth models in U-Th dating of Speleothems. Quaternary Science Reviews, 369, p.109626.
