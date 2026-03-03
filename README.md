## Install 
pip install ibis-uth

## Dependencies
Dependencies are written in both requirements.txt and pyproject.poml. <br>
pip install should import all neccessary dependencies however python verion >= 3.10 is needed

## Import

import ibis <br>
from ibis.IBIS_Main import IBIS

## IBIS
-----

IBIS is a probabilisitc Bayesian model to infer unique sample-specific initial thorium compositions estaimtes thoughtout a speleothem and "correct" apparent U—Th ages accordingly. 

## Why does this matter?
-----
* Ultimately the precision and accuracy of the U—Th method is dependent on this correct.
* Correction can bias ages dramatically, in particular for "dirtier" samples higher thorium content - low measured $^{230}$Th/$^{232}$Th activity ratios and also young samples with a lower "true" $^{230}$Th radiogenic budget.


## Age Equation


$$  \Delta_{\lambda} = \lambda_{230} -  \lambda_{234}$$


$$  \bigg[\bigg(\frac{^{230}Th}{^{238}U}\bigg)_A - \bigg(\frac{^{232}Th}{^{238}U}\bigg)_A \bigg(\frac{^{230}Th}{^{232}Th}\bigg)_{A0} e^{-\lambda_{230}t}\bigg]= 1 - e^{-\lambda_{230}t} + \bigg[ \bigg( \frac{^{234}U}{^{238}U}\bigg)_{A} - 1\bigg]\frac{\lambda_{230}}{\Delta_{\lambda}}(1 - e^{-\Delta_{\lambda}t})$$


* $$\lambda_{230} 230Th decay constant
* $\lambda_{234}$ - 234U decay constant
* $\bigg(\frac{^{230}Th}{^{238}U}\bigg)_A$ - measured activity ratio of 230Th to 238U
*  $\bigg(\frac{^{232}Th}{^{238}U}\bigg)_A$ - measured activity ratio of 232Th to 238U
*  $\bigg(\frac{^{234}U}{^{238}U}\bigg)_A$ - measured activity ratio of 234U to 238U
*  $\bigg(\frac{^{230}Th}{^{232}Th}\bigg)_{A0}$ - estiamte initial activity ratio at the time of formation/deposition/crystallization


A U-Th age requires the initial thorium $\bigg(\frac{^{230}Th}{^{232}Th}\bigg)_{A0}$ correction. In the age equation you need to guess a value for this component at the time of deposition and then it is brought forward in time to the present day. This present day initial is then used to correct the present day measured activity ratios. 

# Fixed vs. variable correction
-----
Common practice in U—Th dating is to use an assumed initial thorium distribution (some mean and some uncertainty) and then apply this uniformly to all samples within a given speleothem. 

#### Bulk Earth
--
An attractive approach is to use the average U-Th content of the upper crust (Taylor and Mclennan, 1995). This gives a thorium activity ratio of 230Th/232Th = 0.83 ± 0.43 (1sigma). The value would then be used to correct all samples using the above age equation framework. 

* Assumption that there is not temporal or spatial variability in $^{230}$Th/$^{232}$Th
* The upper crust is a reasonable description for all karst environments.

#### Hellstrom, (2006) like approach. 
----
Another approach is to use stratigraphic constraints to determine a boutique initial thorium correction that is then used to correct all samples uniformly (Hellstrom, 2006). An extension of this to include co-evality constraints is shown by Roy-Barman and Pons-Branchu, (2010). 

* Similar to Bulk-Earth this is sill a fixed appraoch in that a single distribution is constructed from the speleothem globally and this is applied throughout.
* Better than bulk-earth, as it uses the data and physical stratigraphic constraints to define a "best" global correction but does not account for the potential that the initial thorium incorporation is variable in space and time. 


#### Isochrons
----
The "gold standard" approach to estimation joint age—initial thorium composition is to use an isochron. I term this "gold standard" because it is the only approach that provides a truly independent estiamte with no assumptions. That is not to say that it is always perfect and in some cases far form it. Carolin et al. (2016) provide a suite of isochrons that range in scatter showing the good, bad, and ugly of using isochrons. 

* Enough sample (Ludwig and titterington, ~ 6)
* Enough spread in U-Th ratio.

There is a secondary aspect to this in that a common practice is to measured one (e.g., Carolin et al. 2016) or multipe (e.g., Moseley et al., 2015) and then apply this correction to all samples or bracketing samples throughout the speleothem. This makes it a "fixed" correction as a single initial thorium correction distribution is applied to a series of samples so it is not truly allowed to vary throughout. 

##### References
Carolin, S.A., Cobb, K.M., Lynch-Stieglitz, J., Moerman, J.W., Partin, J.W., Lejau, S., Malang, J., Clark, B., Tuen, A.A. and Adkins, J.F., 2016. Northern Borneo stalagmite records reveal West Pacific hydroclimate across MIS 5 and 6. Earth and Planetary Science Letters, 439, pp.182-193.

Hellstrom, J., 2006. U–Th dating of speleothems with high initial 230Th using stratigraphical constraint. Quaternary geochronology, 1(4), pp.289-295.

Kinsley, C.W., Carter, J.N. and Sharp, W.D., 2025. IBIS: an Integrated Bayesian approach for unique Initial thorium corrections and age-depth models in U-Th dating of Speleothems. Quaternary Science Reviews, 369, p.109626.

Moseley, G.E., Richards, D.A., Smart, P.L., Standish, C.D., Hoffmann, D.L., ten Hove, H. and Vinn, O., 2015. Early–middle Holocene relative sea-level oscillation events recorded in a submerged speleothem from the Yucatán Peninsula, Mexico. The Holocene, 25(9), pp.1511-1521.

Roy-Barman, M. and Pons-Branchu, E., 2016. Improved U–Th dating of carbonates with high initial 230Th using stratigraphical and coevality constraints. Quaternary Geochronology, 32, pp.29-39.

Taylor, S.R. and McLennan, S.M., 1995. The geochemical evolution of the continental crust. Reviews of geophysics, 33(2), pp.241-265.


