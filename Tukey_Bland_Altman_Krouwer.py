## coding: utf-8 
#!/usr/bin/env python2/3
#
# File: Tukey_Bland_Altman_Krouwer.py
# Author: Jordan Van Beeck <jordan.vanbeeck@student.kuleuven.be>

# Description: Module that implements the Tukey's mean difference plots/Bland-Altman plots/Krouwer plots 
# for the statistical comparison of the agreement of two methods.
#
# Publications of interest:
# Giavarina, Understanding Bland Altman analysis, Biochemica Medica, 25(2), pp.141-151, 2015
# http://www.biochemia-medica.com/en/journal/25/2/10.11613/BM.2015.015
# Bland & Altman, Statistical Methods for Assessing Agreement between two Methods, 327(8476), pp. 307-310, 1986
# http://www.sciencedirect.com/science/article/pii/S0140673686908378

#   Copyright 2018 Jordan Van Beeck

#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at

#       http://www.apache.org/licenses/LICENSE-2.0

#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

import numpy as np
import re
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import norm
from scipy import odr
from scipy import stats
from statsmodels.sandbox.regression.predstd import wls_prediction_std
import statsmodels.api as sm
from patsy import dmatrices
import matplotlib.patches as mpatches
import sys
from uncertainties import unumpy

# Make a color dictionary, to be used during plotting
color_dict = { 'Blazhko':'orange', 'RRLyr':'blue'}
patchList = []
for key in color_dict:
        data_key = mpatches.Patch(color=color_dict[key], label=key)
        patchList.append(data_key)


def convert_uncert_df_to_nom_err(df):
    nom_list = []
    err_list = []
    for i in range(len(df)):
        values = df.iloc[i].values
        nom_list.append(unumpy.nominal_values(values))
        err_list.append(unumpy.std_devs(values))
    # generate dataframes containing the nominal values (nom) and the standard deviations (err)
    nom = pd.DataFrame(nom_list,index=list(df.index),columns=list(df))
    err = pd.DataFrame(err_list,index=list(df.index),columns=list(df))
    return nom,err

def f_ODR(B, x):
    # Linear function for orthogonal distance regression: y = m*x + b
    # B is a vector of the parameters.
    # x is an array of the current x values.
    # Return an array in the same format as y, that can be passed to Data or RealData.
    return B[0]*x + B[1]

def generate_fit_models(nom,err,x_variable,weighted):
    # generate design matrix for fit using Patsy
    y,X = dmatrices('Difference ~ ' + str(x_variable) , data=nom, return_type='dataframe')
    # generate a weighted or unweighted linear least squares model using Statsmodels
    if weighted:
        mod = sm.WLS(y,X,weights=1./(err['Difference']**2 + err[str(x_variable)]**2)**(1./2.))
    else:
        mod = sm.OLS(y,X)
    # generate a robust fitting model that punishes outliers, using the Huber loss function
    RLM_mod = sm.RLM(y,X,M=sm.robust.norms.HuberT())
    # generate an orthogonal distance fitting model (Deming regression)
    ODR_mod = odr.Model(f_ODR)
    # generate data format for ODR
    ODR_data = odr.Data(nom[x_variable].values, nom['Difference'].values, 
                        wd=1./np.power(err[x_variable].values,2), 
                        we=1./np.power(err['Difference'].values,2))
    # instantiate ODR with data, model and initial parameter estimate
    odr_mod = odr.ODR(ODR_data, ODR_mod, beta0=[0., 0.]) # initial estimate = 0 * X + 0 = no difference!
    # generate the fit results
    res = mod.fit()
    RLM_res = RLM_mod.fit()
    odr_res = odr_mod.run()
    # Generate a printout of the ODR results (commented out for now)
    # odr_res.pprint()
    # generate residuals and corresponding errors on the residual, to be used in ODR fit plot
    residual_odr,sigma_odr = calc_residual_sigma_odr(odr_res,nom[x_variable].values,
                                                     nom['Difference'].values,
                                                     err[x_variable].values,
                                                     err['Difference'].values)
    # generate the confidence bands of the (non-)weighted model
    prstd, iv_l, iv_u = wls_prediction_std(res)    
    return X,res,mod,RLM_res,RLM_mod,iv_u,iv_l,odr_res,residual_odr,sigma_odr

def calc_residual_sigma_odr(output,x_data,y_data,x_sigma,y_sigma):
    # Calculate initial residuals and adjusted error 'sigma_odr'
    #                 for each data point
    p = output.beta
    delta   = output.delta   # estimated x-component of the residuals
    epsilon = output.eps     # estimated y-component of the residuals
    #    (dx_star,dy_star) are the projections of the errors of x and y onto
    #    the residual line, respectively, i.e. the differences in x & y
    #    between the data point and the point where the orthogonal residual
    #    line intersects the ellipse' created by x_sigma & y_sigma.
    dx_star = ( x_sigma*np.sqrt( ((y_sigma*delta)**2) /
                    ( (y_sigma*delta)**2 + (x_sigma*epsilon)**2 ) ) )
    dy_star = ( y_sigma*np.sqrt( ((x_sigma*epsilon)**2) /
                    ( (y_sigma*delta)**2 + (x_sigma*epsilon)**2 ) ) )
    # calculat the 'total projected uncertainty' based on these projections
    sigma_odr = np.sqrt(dx_star**2 + dy_star**2)
    # residual is positive if the point lies above the fitted curve,
    #             negative if below
    residual_odr = ( np.sign(y_data-f_ODR(p,x_data))
                  * np.sqrt(delta**2 + epsilon**2) )
    return residual_odr,sigma_odr


def Bland_Altman_main(plx_df,method1,method2,weighted = True):
    # loads in the parallax dataframe in order to obtain parameters for
    # a Tukey mean-difference, also called a Bland-Altman plot, or Krouwer plot,
    # in order to better compare the different methods
    
    # generate list of stars  
    stars = list(plx_df)
    # generate list of methods
    methods = list(plx_df.index)
    
    # double PML method comparison = 3 + 3 + 1 + 1 rows
    if len(plx_df) > 6:
        for i in range(0,6):
            for j in range(i+1,6):
                # obtain the names of the different methods (different dereddening methods, for the different PML relations)
                methodname1 = methods[i]
                methodname2 = methods[j]
                # beta = differences of parallaxes, alfa = means of parallaxes, GAIA = gaia parallaxes
                beta = pd.DataFrame(plx_df.iloc[i].values - plx_df.iloc[j].values,index=stars,columns=['Difference'])
                alfa = pd.DataFrame((plx_df.iloc[i].values + plx_df.iloc[j].values)/2.,index=stars,columns=['Mean'])
                GAIA = pd.DataFrame(plx_df.loc['GAIA'].values, index=stars,columns=['Reference'])
                # concatenate the different dataframes into one encompassing dataframe
                plot_df = pd.concat([beta,alfa,GAIA],axis='columns')
                if re.search('(Sesar)',methods[i]):
                    # select only RRab stars, since Sesar relation only applies to those
                    plt_df = plot_df[plx_df.loc['RRAB/RRC']=='RRAB']
                    plt_alfa = alfa[plx_df.loc['RRAB/RRC']=='RRAB']
                    plt_beta = beta[plx_df.loc['RRAB/RRC']=='RRAB'] 
                    plt_plx_df = plx_df.loc[:,plx_df.loc['RRAB/RRC']=='RRAB']
                    # convert df input containing ufloats to readable output
                    nom,err = convert_uncert_df_to_nom_err(plt_df)
                    # Do the actual fitting
                    X,res,mod,RLM_res,RLM_mod,iv_u,iv_l,odr_res,residual_odr,sigma_odr = generate_fit_models(nom,err,'Mean',weighted)
                    # Generate the Bland-Altman/Krouwer plots
                    Bland_Altman_Krouwer_plot(plt_plx_df,nom,err,X,res,iv_u,iv_l,RLM_res,
                                              odr_res,sigma_odr,residual_odr,methodname1,
                                              method2=methodname2)
                    # Generate normality histograms
                    Normality_histogram(nom['Difference'],methodname1,method2=methodname2)
                    # Generate normality assessments
                    Normality_tests(nom['Difference'],methodname1)
                    Normality_tests(nom['Difference'],methodname2)
                    # Generate regression diagnostics plots
                    regression_diagnostics_plot(mod,X,plt_alfa,plt_beta,plt_plx_df,'Mean',weighted,'Tukey')
                else:
                    # convert df input containing ufloats to readable output
                    nom,err = convert_uncert_df_to_nom_err(plot_df)
                    # Do the actual fitting
                    X,res,mod,RLM_res,RLM_mod,iv_u,iv_l,odr_res,residual_odr,sigma_odr = generate_fit_models(nom,err,'Mean',weighted)
                    # Generate the Bland-Altman/Krouwer plots
                    Bland_Altman_Krouwer_plot(plx_df,nom,err,X,res,iv_u,iv_l,RLM_res,
                                              odr_res,sigma_odr,residual_odr,methodname1,
                                              method2=methodname2)
                    # Generate normality histograms
                    Normality_histogram(nom['Difference'],methodname1,method2=methodname2)
                    # Generate normality assessments
                    Normality_tests(nom['Difference'],methodname1)
                    Normality_tests(nom['Difference'],methodname2)
                    # Generate regression diagnostics plots
                    regression_diagnostics_plot(mod,X,alfa,beta,plx_df,'Mean',weighted,'Tukey')
        return
          
    # PML + GAIA method comparison = 3 + 1 + 1 + 1 rows
    else:
        for i in range(3):
            # obtain method name (distinction between dereddening)
            methodname = methods[i]
            # beta = differences of parallaxes, alfa = means of parallaxes, alfa_Krouwer = gaia parallax
            beta = pd.DataFrame(plx_df.loc['GAIA'].values - plx_df.iloc[i].values, index=stars,columns=['Difference'])
            alfa_Krouwer = pd.DataFrame(plx_df.loc['GAIA'].values, index=stars,columns=['Reference'])
            alfa = pd.DataFrame((plx_df.loc['GAIA'].values + plx_df.iloc[i].values)/2., index=stars,columns=['Mean'])
            # concatenate the different dataframes into one encompassing dataframe
            plot_df = pd.concat([beta,alfa_Krouwer,alfa],axis='columns')
            if re.search('(Sesar)',methods[i]):
                # select only RRab stars, since Sesar relation only applies to those
                plt_df = plot_df[plx_df.loc['RRAB/RRC']=='RRAB']
                plt_alfa = alfa[plx_df.loc['RRAB/RRC']=='RRAB']
                plt_beta = beta[plx_df.loc['RRAB/RRC']=='RRAB'] 
                plt_plx_df = plx_df.loc[:,plx_df.loc['RRAB/RRC']=='RRAB']
                plt_alfa_Krouwer = alfa_Krouwer[plx_df.loc['RRAB/RRC']=='RRAB']
                # convert df input containing ufloats to readable output
                nom,err = convert_uncert_df_to_nom_err(plt_df)
                # Do the actual fitting
                X,res,mod,RLM_res,RLM_mod,iv_u,iv_l,odr_res,residual_odr,sigma_odr = generate_fit_models(nom,err,'Mean',weighted)
                X_K,res_K,mod_K,RLM_res_K,RLM_mod_K,iv_u_K,iv_l_K,odr_res_K,residual_odr_K,sigma_odr_K = generate_fit_models(nom,err,'Reference',weighted)
                # Generate the Bland-Altman/Krouwer plots
                Bland_Altman_Krouwer_plot(plt_plx_df,nom,err,X_K,res_K,iv_u_K,iv_l_K,
                                          RLM_res_K,odr_res_K,sigma_odr_K,
                                          residual_odr_K,methodname)
                Bland_Altman_Krouwer_plot(plt_plx_df,nom,err,X,res,iv_u,iv_l,RLM_res,
                                          odr_res,sigma_odr,residual_odr,methodname,method2=method2)
                # Generate normality histograms
                Normality_histogram(nom['Difference'],methodname,method2='GAIA Reference')
                # Generate normality assessments
                Normality_tests(nom['Difference'],methodname)
                # Generate regression diagnostics plots                  
                regression_diagnostics_plot(mod,X,plt_alfa,plt_beta,plt_plx_df,'Mean',weighted,'Tukey')
                regression_diagnostics_plot(mod_K,X_K,plt_alfa_Krouwer,plt_beta,plt_plx_df,'Reference',weighted,'Krouwer')
            else:
                # convert df input containing ufloats to readable output
                nom,err = convert_uncert_df_to_nom_err(plot_df)
                # Do the actual fitting
                X,res,mod,RLM_res,RLM_mod,iv_u,iv_l,odr_res,residual_odr,sigma_odr = generate_fit_models(nom,err,'Mean',weighted)
                X_K,res_K,mod_K,RLM_res_K,RLM_mod_K,iv_u_K,iv_l_K,odr_res_K,residual_odr_K,sigma_odr_K = generate_fit_models(nom,err,'Reference',weighted)
                # Generate the Bland-Altman/Krouwer plots
                Bland_Altman_Krouwer_plot(plx_df,nom,err,X_K,res_K,iv_u_K,iv_l_K,
                                          RLM_res_K,odr_res_K,sigma_odr_K,
                                          residual_odr_K,methodname)
                Bland_Altman_Krouwer_plot(plx_df,nom,err,X,res,iv_u,iv_l,RLM_res,
                                          odr_res,sigma_odr,residual_odr,methodname,method2=method2)
                # Generate normality histograms
                Normality_histogram(nom['Difference'],methodname,method2='GAIA Reference')
                # Generate normality assessments
                Normality_tests(nom['Difference'],methodname)
                # Generate regression diagnostics plots                  
                regression_diagnostics_plot(mod,X,alfa,beta,plx_df,'Mean',weighted,'Tukey')
                regression_diagnostics_plot(mod_K,X_K,alfa_Krouwer,beta,plx_df,'Reference',weighted,'Krouwer')
        return 

def regression_diagnostics_plot(mod,X,plotalfa,plotbeta,df,x_string,weighted,plotstring):
    # get nominal values
    alfa,_ = convert_uncert_df_to_nom_err(plotalfa)
    beta,_ = convert_uncert_df_to_nom_err(plotbeta)
    # if weighted fit, convert the fitted weighted parameters to a format that is readible for the diagnostics plot tools (so that one displays the diagnostics for the weighted parameters)
    if weighted:
        res_infl_mod = sm.OLS(pd.DataFrame(mod.wendog,index=list(alfa.index),columns=['Difference']),
                              pd.DataFrame(mod.wexog,index=list(alfa.index),columns=['Intercept',x_string]))
        res_infl = res_infl_mod.fit()
    else:
        res_infl = mod.fit()
    
    fig, axes = plt.subplots(nrows=2, ncols=2)
    # Generate Cook's distance influence plot
    sm.graphics.influence_plot(res_infl, ax=axes[0,0], criterion="cooks") 
    # Generate leverage plot
    sm.graphics.plot_leverage_resid2(res_infl, ax=axes[0,1]) 
    # Generate fit plot
    sm.graphics.plot_fit(res_infl, x_string, ax=axes[1,0]) 
    # Generate residual plot    
    axes[1,1].scatter(alfa.T,beta.T - res_infl.predict(X).T, color=[ color_dict[u] for u in df.loc['Blazhko/RRLyr'] ]) 
    axes[1,1].axhline(color='r')
    axes[1,1].set_xlabel(x_string + " " + r'$\varpi$' + " (mas)")
    axes[1,1].set_ylabel("Residual " + r'$\varpi$' + " (mas)")
    axes[1,1].set_title("Residual Plot (" + plotstring + ")")
    # Generate extra legend displaying whether the RR Lyrae star is Blazhko-modulated or not
    third_legend = plt.legend(handles=patchList, frameon=True, fancybox=True, framealpha=1.0)
    third_frame = third_legend.get_frame()
    third_frame.set_facecolor('White')
    axes[1,1].add_artist(third_legend)
    return

def Normality_histogram(differences,method1,method2='GAIA Reference'):
    # create a histogram of the differences, containing a Kernel Density Estimation and normal distribution fit,
    # in order to assess normality of the differences distribution.
    # (Need normality in this parameter in order to use the prediction/confidence intervals, as well as limits of agreement! See Bland-Altman publications mentioned on top.)
    sns.set_style('darkgrid')
    plt.figure()
    sns.distplot(differences) # histogram + Kernel Density estimation ("KDE")
    sns.distplot(differences,fit=norm,kde=False) # histogram (overlay) + normal distribution fit ("Norm")
    Legend = plt.legend(['Norm', 'KDE'],loc='upper left', frameon=True, fancybox=True, framealpha=1.0);
    frame = Legend.get_frame()
    frame.set_facecolor('White')
    plt.xlabel('Differences between ' + method1 + ' parallax and ' + method2 + ' parallax')
    plt.ylabel('Probability Density')
    return

def Pair_grid(df_list,df,GAIA):
    # use first item/method in dataframe list to obtain the star list
    totdf = df_list[0]
    # create dataframe containing GAIA data
    GAIAdf = pd.DataFrame(GAIA.reshape(1,len(GAIA)),index=["GAIA"],columns=list(totdf))
    # append results of other methods
    for i in range(1,len(df_list)):
        totdf = pd.concat([totdf,df_list[i]])
    # construct the final dataframe containing results of all methods and GAIA
    totdf = pd.concat([totdf,GAIAdf])
    # obtaining partial dataframes consisting of results using different dereddening methods
    script = totdf[~totdf.index.to_series().str.contains('(_SFD)|(_S_F)')] # our dereddening script
    S_F = pd.concat([totdf[totdf.index.to_series().str.contains('(_SFD)')],GAIAdf]) # S_F dust map (have to add GAIA data again)
    SFD = pd.concat([totdf[totdf.index.to_series().str.contains('(_S_F)')],GAIAdf]) # SFD dust map (have to add GAIA data again)
    # Loop over the results list using different dereddening methods, generating the pair grid for each
    for j in [script,S_F,SFD]:
        # get the nominal values of the dataframe containing ufloats
        nom,_ = convert_uncert_df_to_nom_err(j)
        # generate necessary dataframes for pair grid
        sns.set_context("talk")
        transpose_nom = nom.T
        nom_na = transpose_nom.dropna(axis=1)
        transpose_nom['B/RR'] = pd.Series(df.loc['B/RR'].values, index=transpose_nom.index) # add Blazhko/RRLyr distinction
        transpose_nom_nan = transpose_nom.dropna(axis=1) # drop any columns containing NaN values 
        # (i.e. in our case: Sesar relation column, since this is only applicable to RRAB, in principle, if we find other RRAB only relations we can compare those with Sesar!)

        # generate the actual pair grid (for more detailed information, see Seaborn website)
        g = sns.PairGrid(nom_na)
        g.map_upper(plt.scatter,color=[ color_dict[u] for u in transpose_nom_nan['B/RR'] ]) # scatterplots making a coloured distinction between Blazhko and RRLyr
        g.map_lower(sns.kdeplot) # 2D KDE
        g.map_diag(sns.kdeplot, lw=2, legend=False); # 1D KDE on diagonals
        g.add_legend(handles=patchList) # legend containing the Blazkho/RRLyr distinction       
    return

def Normality_tests(differences,method): 
    # print the output of different statistical tests for normality
    shap = stats.shapiro(differences)
    print("Shapiro-Wilk Results Method " + method + ":")
    print(shap)
    print(" ")
    ands = stats.anderson(differences)
    print("Anderson-Darling Results Method " + method + ":")
    print(ands)
    print(" ")
    kolm = stats.kstest(differences, 'norm')
    print("Kolmogorov-Smirnov Results Method " + method + ":")
    print(kolm)
    print(" ")
    if len(differences) > 7:
        dagos = stats.normaltest(differences)
        print("D'agostino-Pearson Results Method " + method + ":")
        print(dagos)
        print(" ")
    else:
        print("No D'agostino-Pearson method possible for method " + method)
        print(" ")
    return

def difference_plot(df_diff,df_e_diff,method1,method2,degreesoffreedom,Critical_t,x_string):
    plt.figure()
    # plot the differences vs. the reference values (Krouwer) or means (Tukey)
    plt.errorbar(df_diff[x_string],df_diff['Difference'],yerr=df_e_diff['Difference'],
                 xerr=df_e_diff[x_string],ls='None', color=[ color_dict[u] for u in df_diff['Blazhko/RRLyr'] ])
    # generate color legend displaying Blazhko/RRlyr difference
    first_legend = plt.legend(handles=patchList, loc=2, frameon=True, fancybox=True, framealpha=1.0)
    first_frame = first_legend.get_frame()
    first_frame.set_facecolor('White')
    ax = plt.gca().add_artist(first_legend)
    # generate plotting space for confidence bounds, limits of agreement, and mean difference
    xspace = np.linspace(df_diff[x_string].values.min()-df_e_diff.iloc[df_diff[x_string].values.argmin()][x_string],
                         df_diff[x_string].values.max()+df_e_diff.iloc[df_diff[x_string].values.argmax()][x_string],
                         num=200)
    # calculate the confidence bounds, limits of agreement and mean difference
    meandiff = df_diff.mean()['Difference']
    e_meandiff = df_diff.std()['Difference'] * np.sqrt(1./degreesoffreedom)
    conf_meandiff = e_meandiff * Critical_t
    LA_up = np.ones(len(xspace))* (meandiff + 1.96 * e_meandiff * np.sqrt(degreesoffreedom))
    LA_down = np.ones(len(xspace))* (meandiff - 1.96 * e_meandiff * np.sqrt(degreesoffreedom))
    conf_LA = e_meandiff * np.sqrt(3) * Critical_t
    # plot the limits of agreement and mean difference
    plt.plot(xspace,np.ones(len(xspace))* meandiff, color='k',ls='-',label='Mean')
    plt.plot(xspace,LA_down,color='red',ls='--',label='LA')
    plt.plot(xspace,LA_up,color='red',ls='--')
    # generate coloured bands which signify the confidence bounds
    ax = plt.gca()
    ax.fill_between(xspace, meandiff-np.ones(len(xspace))*conf_meandiff, meandiff+np.ones(len(xspace))*conf_meandiff, facecolor='green', alpha=0.2)
    ax.fill_between(xspace, LA_up-np.ones(len(xspace))*conf_LA, LA_up+np.ones(len(xspace))*conf_LA, facecolor='green', alpha=0.2)
    ax.fill_between(xspace, LA_down-np.ones(len(xspace))*conf_LA, LA_down+np.ones(len(xspace))*conf_LA, facecolor='green', alpha=0.2)  
    # generate legend
    Legend = plt.legend(frameon=True, fancybox=True, framealpha=1.0)
    frame = Legend.get_frame()
    frame.set_facecolor('White')
    # set title and axis labels
    if x_string=='Reference':
        plt.title('Krouwer Difference plot ' + method1)
        plt.xlabel('GAIA ' + r'$\varpi$')
    else:
        if len(method2) == 0:
            plt.title('Tukey Mean Difference plot ' + method1 + ' and GAIA')
        else:
            plt.title('Tukey Mean Difference plot ' + method1 + ' and ' + method2)
        plt.xlabel(x_string + ' ' + r'$\varpi$')
    plt.ylabel(r'$\varpi$' + ' Difference')    
    return
    
def generate_sorted_df_fit(df_diff,df_e_diff,fit_res,robust_fit_res,dmatrix,predbound_up,predbound_down,odr_res,x_string):
    # generate the different fit prediction values and put them in a dataframe
    OLS_predict = fit_res.predict(dmatrix)
    RLS_predict = robust_fit_res.predict(dmatrix)
    ODR_predict = odr_res.y
    ODR_predict_x = odr_res.xplus
    predict_df = pd.DataFrame(np.vstack((OLS_predict,RLS_predict,ODR_predict,ODR_predict_x)),index=["OLS","RLS","ODR","ODR_x"],columns=list(predbound_up.index))
    # transpose the dataframe, in order to add the prediction bounds for OLS
    predict_df = predict_df.T
    predict_df['pred_up'] = pd.Series(predbound_up.values, index=predict_df.index)
    predict_df['pred_down'] = pd.Series(predbound_down.values, index=predict_df.index)
    # generate a new dataframe containing the Differences, Reference (GAIA) parallaxes, and Mean parallaxes and their errors!  
    newdf_diff = df_diff.copy()
    newdf_diff['e_Difference'] = pd.Series(df_e_diff['Difference'].values, index=predict_df.index)
    newdf_diff['e_Reference'] = pd.Series(df_e_diff['Reference'].values, index=predict_df.index)
    newdf_diff['e_Mean'] = pd.Series(df_e_diff['Mean'].values, index=predict_df.index)
    # merge the predict dataframe and the difference dataframe
    mergeddf = pd.concat([newdf_diff,predict_df],axis=1)
    # sort the merged dataframe
    sorteddf = mergeddf.sort_values(x_string)
    return sorteddf


def fit_plot(sorteddf,odr_res,sigma_odr,residual_odr,x_string,method,additional_method=''):
    # set the plotting style
    sns.set_style('darkgrid')
    ODR_fit_plot(sorteddf[x_string],sorteddf['Difference'],sorteddf['e_'+x_string],
                 sorteddf['e_Difference'],odr_res,sigma_odr,residual_odr,x_string,
                 method,additional_method=additional_method)
    plt.figure()
    ax = plt.gca()
    # plot the differences in function of either reference (Krouwer) or mean (Tukey) parallaxes
    plt.scatter(sorteddf[x_string],sorteddf['Difference'], color=[ color_dict[u] for u in sorteddf['Blazhko/RRLyr'] ])
    # add legend containing the distinction between Blazhko/RRLyr
    second_legend = plt.legend(handles=patchList, loc=2,frameon=True, fancybox=True, framealpha=1.0)
    second_frame = second_legend.get_frame()
    second_frame.set_facecolor('White')
    ax.add_artist(second_legend)
    
    # plot the fits of the differences in function of either reference (Krouwer) or mean (Tukey) parallaxes
    plt.plot(sorteddf[x_string],sorteddf['OLS'],label='OLS Fit')
    plt.plot(sorteddf[x_string],sorteddf['RLS'],label='RLS Fit')
    plt.plot(sorteddf['ODR_x'],sorteddf['ODR'],label='ODR Fit')
    plt.plot(sorteddf[x_string],sorteddf['pred_up'],c='r',ls='--',label='Prediction Bound')
    plt.plot(sorteddf[x_string],sorteddf['pred_down'],c='r',ls='--',label='')
    # generate the correct labels
    if x_string=='Reference':
        plt.title('Fitted ' + r'$\varpi$' + ' Differences ' + '(Krouwer, ' + method +')')
        plt.xlabel('GAIA ' + r'$\varpi$')
    else:
        if len(additional_method) == 0:
            plt.title('Fitted ' + r'$\varpi$' + ' Differences ' + '(Tukey, ' + method + ', GAIA)')
        else:
            plt.title('Fitted ' + r'$\varpi$' + ' Differences ' + '(Tukey, ' + method + ', ' + additional_method + ')')
        plt.xlabel(x_string + ' ' + r'$\varpi$')
    plt.ylabel(r'$\varpi$' + ' Difference')
    # add the legend containing the distinction between the different fits
    Legend2 = plt.legend(loc=1,frameon=True, fancybox=True, framealpha=1.0)
    frame2= Legend2.get_frame()
    frame2.set_facecolor('White')
    return

def ODR_fit_plot(x_data,y_data,x_sigma,y_sigma,output,sigma_odr,residual_odr,x_string,method,additional_method=''):
    # create figure
    fig = plt.figure()
    # Make the subplot
    fit = fig.add_subplot(211)
    # remove tick labels from upper plot
    fit.set_xticklabels( () )
    # set y-label
    plt.ylabel(r'$\varpi$' + ' Difference')
    # set title
    if x_string=='Reference':
        plt.title('Fitted (ODR) ' + r'$\varpi$' + ' Differences ' + '(Krouwer, ' + method +')')
    else:
        if len(additional_method) == 0:
            plt.title('Fitted (ODR) ' + r'$\varpi$' + ' Differences ' + '(Tukey, ' + method + ', GAIA)')
        else:
            plt.title('Fitted (ODR) ' + r'$\varpi$' + ' Differences ' + '(Tukey, ' + method + ', ' + additional_method + ')')
    # Generate linspace for plotting
    stepsize = (max(x_data)-min(x_data))/1000.
    # set plotting margin to 50 times the stepsize (beyond and before max and min, respectively)
    margin = 50*stepsize
    x_model = np.arange( min(x_data)-margin,max(x_data)+margin,
                                    stepsize)
    # plot the ODR fit
    fit.plot(x_data,y_data,'ro', x_model, f_ODR(output.beta,x_model),markersize=4,label='')
    # Add error bars
    fit.errorbar(x_data, y_data, xerr=x_sigma, yerr=y_sigma, fmt='r+', label='Differences')
    # set y-scale to linear
    fit.set_yscale('linear')
    # draw starting guess (in our case just the x-axis) as dashed green line
    fit.axhline(y=0, c='g',linestyle="-.", label="No Diff")
    
    # output.xplus = x + delta
    a = np.array([output.xplus,x_data])
    # output.y = f(p, xfit), or y + epsilon
    b = np.array([output.y,y_data])
    # plot the actual fit
    fit.plot( a[0][0], b[0][0], 'b-', label= 'Fit')
    # plot the residuals
    fit.plot(np.array([a[0][0],a[1][0]]),np.array([b[0][0],b[1][0]]),'k--', label = 'Residuals')
    for i in range(1,len(y_data)):
        fit.plot( a[0][i], b[0][i], 'b-')
        fit.plot( np.array([a[0][i],a[1][i]]),np.array([b[0][i],b[1][i]]),'k--')
    # plot the legend
    legend = fit.legend(frameon=True, fancybox=True, framealpha=1.0)
    frame= legend.get_frame()
    frame.set_facecolor('White')
    
    # separate plot to show residuals
    residuals = fig.add_subplot(212)
    residuals.errorbar(x=x_data,y=residual_odr,yerr=sigma_odr,fmt='r+',label = "Residuals")
    # make sure residual plot has same x axis as fit plot
    residuals.set_xlim(fit.get_xlim())
    # Draw a horizontal line at zero on residuals plot
    plt.axhline(y=0, color='g')
    # Label axes
    if x_string=='Reference':
        plt.xlabel('GAIA ' + r'$\varpi$')
    else:
        plt.xlabel(x_string + ' ' + r'$\varpi$')
    # set a plain tick-label style
    plt.ticklabel_format(style='plain', useOffset=False, axis='x')
    plt.ylabel('Residual ' + r'$\varpi$' + ' Difference')
    return

def Bland_Altman_Krouwer_plot(df_plx,df_diff,df_e_diff,dmatrix,fit_res,predbound_up,predbound_down,robust_fit_res,odr_res,sigma_odr,residual_odr,method1,method2=False):
    # using the Bland_Altman beta's and alfa's previously calculated, the Bland-Altman plot or the Krouwer plot is generated,
    # as well as returning limits of agreement, with their 95% confidence intervals.

    df_diff['Blazhko/RRLyr'] = pd.Series(df_plx.loc['Blazhko/RRLyr'].values, index=df_diff.index)
    df_e_diff['Blazhko/RRLyr'] = pd.Series(df_plx.loc['Blazhko/RRLyr'].values, index=df_e_diff.index)

    # generate the degrees of freedom, as well as the critical value of the student's t distribution
    degreesoffreedom = df_diff.shape[0]
    p = 0.95
    Critical_t = stats.t.ppf(p, degreesoffreedom-1)

    if  isinstance(method2, basestring):        
        # Generate the Tukey plot
        difference_plot(df_diff,df_e_diff,method1,method2,degreesoffreedom,Critical_t,'Mean')        
        # Make a sorted df containing the information needed for a fit plot
        sorteddf = generate_sorted_df_fit(df_diff,df_e_diff,fit_res,robust_fit_res,
                                          dmatrix,predbound_up,predbound_down,odr_res,'Mean')
        # Make new figure showing the fits to the differences
        fit_plot(sorteddf,odr_res,sigma_odr,residual_odr,'Mean',method1,additional_method=method2)  
    else:
        # Generate the Krouwer plot
        difference_plot(df_diff,df_e_diff,method1,method2,degreesoffreedom,Critical_t,'Reference')        
        # Make a sorted df containing the information needed for a fit plot
        sorteddf = generate_sorted_df_fit(df_diff,df_e_diff,fit_res,robust_fit_res,
                                          dmatrix,predbound_up,predbound_down,odr_res,'Reference')
        # Make new figure showing the fits to the differences
        fit_plot(sorteddf,odr_res,sigma_odr,residual_odr,'Reference',method1)  
    return 
