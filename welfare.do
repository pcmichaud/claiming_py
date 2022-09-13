clear all
capture log close
set more off 

capture cd ~/cedia/projets/rsi-05-claiming/estimation

import delimited data/estimation_sample_welfare.csv, clear

global x "healthproblems age married educ finlit retlit"
keep respid q37_* npv_loss_* treatment frame $x

reshape long q37_ npv_loss_ npv_loss_rel_, i(respid) j(scn)

drop if npv_loss_>=100e3

gen treat_nra = frame==2 if inlist(scn,1,7)
gen treat_bke = treatment=="Break-even" if treatment!="Insurance" & scn!=7
gen treat_ins = treatment=="Insurance" if treatment!="Break-even" & scn!=7

gen post_nra = scn==7
gen post_edu = scn>=2

gen post_treat_nra = treat_nra*post_nra
gen post_treat_bke = post_edu*treat_bke
gen post_treat_ins = post_edu*treat_ins

* dollars
global x "healthproblems age married finlit retlit"
reg npv_loss_ $x post_nra treat_nra post_treat_nra, cluster(respid)
estimates store d_frame
reg npv_loss_ $x post_edu treat_bke post_treat_bke, cluster(respid)
estimates store d_bke
reg npv_loss_ $x post_edu treat_ins post_treat_ins, cluster(respid)
estimates store d_ins

esttab d_frame d_bke d_ins, se 


reg npv_loss_rel_ $x post_nra treat_nra post_treat_nra, cluster(respid)
estimates store r_frame
reg npv_loss_rel_ $x post_edu treat_bke post_treat_bke, cluster(respid)
estimates store r_bke
reg npv_loss_rel_ $x post_edu treat_ins post_treat_ins, cluster(respid)
estimates store r_ins

esttab r_frame r_bke r_ins, se 
