import numpy as np
import pandas as pd
import scipy
import json
from sklearn.linear_model import LinearRegression

import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning) 
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


class BiasCorrection:

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def _warning(self, msg, prefix="WARNING"):
        print(prefix, '-', msg)
        # exit()

    def _error(self, msg, prefix="ERROR"):
        print(prefix, '-', msg)
        exit()

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def _load_dataframe(self, d, specifier=None):
        if(isinstance(d, str)):
            try:
                df = pd.read_csv(d)
            except FileNotFoundError:
                self._error(f"Could not find {specifier+' ' if specifier is not None else ''}file: {d}")
            except pd.errors.ParserError:
                self._error(f"Could not parse {specifier+' ' if specifier is not None else ''}file (.csv expected): {d}")
        elif(isinstance(d, pd.DataFrame)):
            df = d
        return df

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def _load_config(self, d, specifier=None):
        if(isinstance(d, str)):
            try:
                with open(d) as cfg_json:
                    cfg = json.load(cfg_json)
            except FileNotFoundError:
                self._error(f"Could not find {specifier+' ' if specifier is not None else ''}file: {d}")
            except json.decoder.JSONDecodeError:
                self._error(f"Could not parse {specifier+' ' if specifier is not None else ''}file (.json expected): {d}")
        elif(isinstance(d, pd.DataFrame)):
            cfg = d
        return cfg

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def __init__(self, counts=None, samples=None, variants=None, config=None, outdir=None,
                       init_bias_prev=None, init_bias_susc=None, init_normalization_factors=None,
                       pseudocount=0.5, output_intermediates=True, output_final=True, seed=None):

        if(seed is not None):
            np.random.seed(seed)
            self.seed = seed

        self.log    = {}
        self.outdir = outdir
        
        self.output_intermediates = output_intermediates
        self.output_final = output_final

        #------------------------------
        # Load counts data and metadata for samples and variants:
        self.cfg          = self._load_config(config)
        self.samplesInfo  = self._load_dataframe(samples)
        self.variantsInfo = self._load_dataframe(variants)
        self.counts       = self._load_dataframe(counts)

        #------------------------------
        # Apply pseudocount to zero counts
        self.counts[self.counts == 0] += pseudocount

        #------------------------------
        # Assign regression weights to counts:
        self.weights = np.minimum(self.counts, self.cfg['regression_weight_saturation_count'])

        #------------------------------
        # Determine which data are considered 'trustworthy' for bias inference:
        self.trustworthy = self.get_trustworthy_variants(min_trustworthy_count=300, drop_top=100)

        #------------------------------
        # Instantiate dataframes for fitnesses, intercepts, and residuals returned from log-linear fits
        self.fitnesses       = pd.DataFrame(np.nan, index=self.counts.index, columns=self.assays)
        self.intercepts      = pd.DataFrame(np.nan, index=self.counts.index, columns=self.assays)
        self.residuals       = pd.DataFrame(np.nan, index=self.counts.index, columns=self.counts.columns)
        self.adjusted_counts = pd.DataFrame(np.nan, index=self.counts.index, columns=self.counts.columns)

        #------------------------------
        # Initialize bias terms to be optimized:
        if(init_bias_susc is not None):
            self.variantsInfo['bias_susceptibility'] = init_bias_susc
        else:
            self.variantsInfo['bias_susceptibility'] = ( np.random.normal(loc=self.cfg['init_bias_susceptibility_mean'], scale=self.cfg['init_bias_susceptibility_stdev'], size=len(self.variantsInfo)) 
                                                        if self.cfg['init_bias_susceptibility_stdev'] > 0 else self.cfg['init_bias_susceptibility_mean'] )
        #----------
        if(init_bias_prev is not None):
            self.samplesInfo['bias_prevalence'] = init_bias_prev
        else:
            self.samplesInfo['bias_prevalence'] = ( np.random.normal(loc=self.cfg['init_bias_prevalence_mean'], scale=self.cfg['init_bias_prevalence_stdev'], size=len(self.samplesInfo))
                                                    if self.cfg['init_bias_prevalence_stdev'] > 0 else self.cfg['init_bias_prevalence_mean'] )
        #------------------------------
        # Initialize depth normalization_factors for each sample:
        if(init_normalization_factors is not None):
            self.samplesInfo['normalization_factor'] = init_normalization_factors
        else:
            self.samplesInfo['normalization_factor'] = np.nan
            for assay in self.assays:
                samples_a = self.get_assay_samples(assay, exclude_timepts=self.cfg['exclude_timepts'])
                self.samplesInfo.loc[(self.samplesInfo['sample'].isin(samples_a)), 'normalization_factor'] = self.counts.loc[self.trustworthy[assay].values, samples_a].mean().values

        #------------------------------
        # Perform initial linear fits on initial observed counts,
        # to obtain pre-correction fitness estimates and residuals.
        self.perform_fits(min_mean_count=25, verbose=False)
        self.get_observed_counts(update_consensus=True)
        self.get_adjusted_counts(update_consensus=True)
        self.save(label='initial', save_to_file=self.output_intermediates)

        
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


    @property
    def assays(self):
        return self.samplesInfo['assay'].unique()

    @property
    def samples(self):
        return self.samplesInfo['sample'].unique()

    @property
    def variants(self):
        return list(self.variantsInfo.index)

    def get_assay_samples(self, assays=None, exclude_timepts=None):
        assays          = self.assays if assays is None else np.atleast_1d(assays)
        exclude_timepts = self.cfg['exclude_timepts'] if exclude_timepts is None else exclude_timepts
        samples         = self.samplesInfo.loc[(self.samplesInfo['assay'].isin(assays)) & (~self.samplesInfo['timept'].isin(exclude_timepts))]['sample'].values
        return samples

    def set_bias_susceptibility(self, bias_susc, variants=None):
        variants = self.variants if variants is None else np.atleast_1d(variants)
        self.variantsInfo.loc[variants, 'bias_susceptibility'] = bias_susc


    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def save(self, label, save_to_file=True):
        self.log[label] = { 'samplesInfo':     self.samplesInfo.copy(),
                            'variantsInfo':    self.variantsInfo.copy(),
                            'fitnesses':       self.fitnesses.copy(), 
                            'intercepts':      self.intercepts.copy(), 
                            'residuals':       self.residuals.copy(),
                            'raw_counts':      self.counts.copy(),
                            'observed_counts': self.observedCounts.copy(),
                            'adjusted_counts': self.adjustedCounts.copy() } 

        if(save_to_file):
            if(self.outdir is None):
                self._warning("No output directory was provided, cannot save data to file.")
            else:
                self.samplesInfo.to_csv(self.outdir+f"/samples_{label}.csv", index=False)
                self.variantsInfo.to_csv(self.outdir+f"/variants_{label}.csv", index=False)
                self.fitnesses.to_csv(self.outdir+f"/fitnesses_{label}.csv", index=False)
                self.intercepts.to_csv(self.outdir+f"/intercepts_{label}.csv", index=False)
                self.residuals.to_csv(self.outdir+f"/residuals_{label}.csv", index=False)
                self.counts.to_csv(self.outdir+f"/counts_{label}.csv", index=False)
                self.observedCounts.to_csv(self.outdir+f"/observedCounts_{label}.csv", index=False)
                self.adjustedCounts.to_csv(self.outdir+f"/adjustedCounts_{label}.csv", index=False)
                self.trustworthy.to_csv(self.outdir+f"/trustworthy_{label}.csv", index=False)                       


    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def get_trustworthy_variants(self, assays=None, exclude_timepts=None, min_trustworthy_count=None, max_trustworthy_count=None, drop_top=None, drop_bottom=None):
        assays          = self.assays if assays is None else np.atleast_1d(assays)
        exclude_timepts = self.cfg['exclude_timepts'] if exclude_timepts is None else exclude_timepts
        min_trustworthy_count = self.cfg['min_trustworthy_count'] if (min_trustworthy_count is None and drop_bottom is None) else min_trustworthy_count if min_trustworthy_count is not None else 0
        max_trustworthy_count = self.cfg['max_trustworthy_count'] if (max_trustworthy_count is None and drop_top is None) else max_trustworthy_count if max_trustworthy_count is not None else np.inf
        #--------------
        trustworthiness = pd.DataFrame(np.nan, index=self.counts.index, columns=assays)
        variantNumTrustworthyTimeSeries = np.zeros(len(self.variantsInfo))
        for assay in assays:
            assayInfo = self.samplesInfo.loc[(self.samplesInfo['assay'] == assay) & (~self.samplesInfo['timept'].isin(self.cfg['exclude_timepts']))].sort_values(by='timept')
            samples_a = assayInfo['sample'].unique()
            mean_counts_a = self.counts[samples_a].mean(axis=1).values
            min_trustworthy_count_a = max(min_trustworthy_count, np.partition(mean_counts_a, (drop_bottom-1))[(drop_bottom-1)]) if (drop_bottom is not None and drop_bottom != 0) else min_trustworthy_count
            max_trustworthy_count_a = min(max_trustworthy_count, np.partition(mean_counts_a, -drop_top)[-drop_top]) if (drop_top is not None and drop_top != 0) else max_trustworthy_count
            variant_trustworthyStatus = (mean_counts_a > min_trustworthy_count_a) & (mean_counts_a < max_trustworthy_count_a) # TODO put inequalities back to <= and >= after replciating mikhail's results
            trustworthiness[assay] = variant_trustworthyStatus
            #----------
            variantNumTrustworthyTimeSeries += variant_trustworthyStatus.astype(int)
        #--------------
        trustworthiness['bias_susceptibility'] = variantNumTrustworthyTimeSeries >= self.cfg['min_num_trustworthy_assays']
        #--------------
        return trustworthiness


    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def initialize_normalization_factors(self, assays=None, exclude_timepts=None):
        assays          = self.assays if assays is None else np.atleast_1d(assays)
        exclude_timepts = self.cfg['exclude_timepts'] if exclude_timepts is None else exclude_timepts
        #--------------
        for assay in assays:
            samples_a = self.get_assay_samples(assay, exclude_timepts=exclude_timepts)
            self.samplesInfo.loc[(self.samplesInfo['sample'].isin(samples_a)), 'normalization_factor'] = self.counts.loc[self.trustworthy[assay].values, samples_a].mean().values


    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def linear_regression(self, x, y, w=None, verbose=False):
        if(verbose):
            print("&&&", "x", x, "y", y, "w", w)
        xw = w*x if w is not None else x
        yw = w*y if w is not None else y
        if(not np.any(yw)):
            slope     = 0
            intercept = 0
        else:
            n = len(x)
            xw_mean = np.sum(xw)/n
            yw_mean = np.sum(yw)/n
            w_mean  = np.sum(w)/n if w is not None else 1
            slope     = np.sum(xw * (y - yw_mean/w_mean)) / np.sum(xw * (x - xw_mean/w_mean))
            intercept = (1/w_mean)*(yw_mean - slope*xw_mean)
        return slope, intercept


    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def get_observed_counts(self, assays=None, variants=None, update_consensus=False):

        variants = self.variants if variants is None else np.atleast_1d(variants)
        assays   = self.assays if assays is None else np.atleast_1d(assays)
        samples  = self.get_assay_samples(assays)

        observedCounts = pd.DataFrame(np.nan, index=self.counts.index, columns=self.counts.columns)

        for a, assay in enumerate(assays):

            assayInfo  = self.samplesInfo.loc[(self.samplesInfo['assay'] == assay) & (~self.samplesInfo['timept'].isin(self.cfg['exclude_timepts']))].sort_values(by='timept')
            samples_a  = assayInfo['sample'].unique()
            timepts_a  = assayInfo['timept'].unique()
            
            normalization_factors_a = assayInfo['normalization_factor'].values

            counts_a   = self.counts.loc[variants, samples_a].values

            observedCounts_a = np.empty((len(variants), len(timepts_a)))*np.nan

            for i, variant in enumerate(variants):
                counts_a_i = counts_a[i]
                
                observedCounts_a_i = counts_a_i/normalization_factors_a

                observedCounts_a[i] = observedCounts_a_i

            observedCounts.loc[variants, samples_a] = observedCounts_a

        if(update_consensus):
            self.observedCounts = observedCounts

        return observedCounts


    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def get_adjusted_counts(self, assays=None, variants=None, update_consensus=False):

        variants = self.variants if variants is None else np.atleast_1d(variants)
        assays   = self.assays if assays is None else np.atleast_1d(assays)
        samples  = self.get_assay_samples(assays)

        bias_susc = self.variantsInfo.loc[variants, 'bias_susceptibility'].values

        adjustedCounts = pd.DataFrame(np.nan, index=self.counts.index, columns=self.counts.columns)

        for a, assay in enumerate(assays):

            assayInfo  = self.samplesInfo.loc[(self.samplesInfo['assay'] == assay) & (~self.samplesInfo['timept'].isin(self.cfg['exclude_timepts']))].sort_values(by='timept')
            samples_a  = assayInfo['sample'].unique()
            timepts_a  = assayInfo['timept'].unique()
            
            normalization_factors_a = assayInfo['normalization_factor'].values
            biasprev_a = assayInfo['bias_prevalence'].values

            counts_a   = self.counts.loc[variants, samples_a].values

            adjustedCounts_a = np.empty((len(variants), len(timepts_a)))*np.nan

            for i, variant in enumerate(variants):
                counts_a_i = counts_a[i]
                biassusc_i = bias_susc[i] 
                
                adjLogCounts_a_i = np.log( counts_a_i/normalization_factors_a * np.exp(biassusc_i * biasprev_a))

                adjustedCounts_a[i] = np.exp(adjLogCounts_a_i)

            adjustedCounts.loc[variants, samples_a] = adjustedCounts_a

        if(update_consensus):
            self.adjustedCounts = adjustedCounts

        return adjustedCounts


    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def perform_fits(self, counts=None, weights=None, assays=None, variants=None, normalization_factors=None, bias_prev=None, bias_susc=None, 
                        min_mean_count=None, update_consensus=True, return_results=False, trustworthy_only=False, verbose=False): 

        normalization_factors  = np.array(normalization_factors) if normalization_factors is not None else None
        bias_prev = np.array(bias_prev) if bias_prev is not None else None
        bias_susc = np.array(bias_susc) if bias_susc is not None else None

        min_mean_count = min_mean_count if min_mean_count is not None else 0

        if(counts is not None): # If a matrix of counts is provided, perform fits on those count vectors and ignore assays/variants arguments.

            slopes_all     = np.empty( (len(counts), 1) )*np.nan                
            intercepts_all = np.empty( (len(counts), 1) )*np.nan                
            residuals_all  = np.empty( (len(counts), counts.shape[1] if isinstance(counts, np.ndarray) else max([len(ts) for ts in counts])) )*np.nan   # pd.DataFrame(np.nan, index=variants, columns=assays)

            for ts, timeseries in enumerate(counts):

                counts_ts  = counts[ts]
                if(np.mean(counts_ts) < min_mean_count):
                    continue
                weights_ts = weights[ts]
                timepts_ts = np.arange(0, len(counts_ts))

                normalization_factors_ts = 1 if normalization_factors  is None else normalization_factors  if (normalization_factors.ndim < 2 and not normalization_factors.dtype is np.dtype('O'))   else normalization_factors[ts]
                biasprev_ts = 0 if bias_prev is None else bias_prev if (bias_prev.ndim < 2 and not bias_prev.dtype is np.dtype('O')) else bias_prev[ts]
                biassusc_ts = 0 if bias_susc is None else bias_susc if bias_susc.ndim < 1 else bias_susc[ts]

                adjLogCounts_ts = np.log( counts_ts/normalization_factors_ts * np.exp(biassusc_ts * biasprev_ts)) # , where=((counts_ts>0)&(normalization_factors_ts!=0)), out=np.zeros_like(counts_ts) )

                slope_ts, intercept_ts = self.linear_regression(x=timepts_ts, y=adjLogCounts_ts, w=weights_ts)
                predLogCounts_ts       = intercept_ts + slope_ts * timepts_ts
                residuals_ts           = adjLogCounts_ts - predLogCounts_ts   

                slopes_all[ts]     = slope_ts
                intercepts_all[ts] = intercept_ts
                residuals_all[ts, 0:len(residuals_ts)] = residuals_ts

            if(update_consensus):
                self._warning("Can't update consensus when perform_fits() is called with the counts=<...> argument. To update consensus, call with assays=<...> and/or variants=<...> arguments or with no arguments to update all assays and variants.")

            return (slopes_all, intercepts_all, residuals_all) if return_results else None

        else: # If no counts are provided, perform fits on count vectors for specified assays and/or variants.
            
            variants = self.variants if variants is None else np.atleast_1d(variants)
            assays   = self.assays if assays is None else np.atleast_1d(assays)
            samples  = self.get_assay_samples(assays)

            bias_susc = self.variantsInfo.loc[variants, 'bias_susceptibility'].values if bias_susc is None else bias_susc

            if(update_consensus):
                self.fitnesses       = pd.DataFrame(np.nan, index=self.counts.index, columns=self.assays)
                self.intercepts      = pd.DataFrame(np.nan, index=self.counts.index, columns=self.assays)
                self.residuals       = pd.DataFrame(np.nan, index=self.counts.index, columns=self.counts.columns)

            if(return_results):
                slopes_all     = np.empty((len(variants), len(assays)))*np.nan             
                intercepts_all = np.empty((len(variants), len(assays)))*np.nan             
                residuals_all  = np.empty((len(variants), len(samples)))*np.nan
                sample_col_counter = 0

            for a, assay in enumerate(assays):

                assayInfo  = self.samplesInfo.loc[(self.samplesInfo['assay'] == assay) & (~self.samplesInfo['timept'].isin(self.cfg['exclude_timepts']))].sort_values(by='timept')
                samples_a  = assayInfo['sample'].unique()
                timepts_a  = assayInfo['timept'].unique()
                
                normalization_factors_a = assayInfo['normalization_factor'].values if normalization_factors is None else normalization_factors if normalization_factors.ndim < 2 else normalization_factors[a]
                biasprev_a = assayInfo['bias_prevalence'].values if bias_prev is None else bias_prev if bias_prev.ndim < 2 else bias_prev[a]

                counts_a   = self.counts.loc[variants, samples_a].values
                weights_a  = self.weights.loc[variants, samples_a].values if weights is not None else np.ones_like(counts_a)

                slopes_a         = np.empty((len(variants), 1))*np.nan             
                intercepts_a     = np.empty((len(variants), 1))*np.nan             
                residuals_a      = np.empty((len(variants), len(timepts_a)))*np.nan

                for i, variant in enumerate(variants):

                    if(trustworthy_only and not self.trustworthy.loc[variant, assay]):
                        continue

                    counts_a_i  = counts_a[i]
                    weights_a_i = weights_a[i]
                    biassusc_i  = bias_susc[i] 

                    adjLogCounts_a_i = np.log( counts_a_i/normalization_factors_a * np.exp(biassusc_i * biasprev_a)) 
                        
                    slope_a_i, intercept_a_i = self.linear_regression(x=timepts_a, y=adjLogCounts_a_i, w=weights_a_i)
                    predLogCounts_a_i        = intercept_a_i + slope_a_i * timepts_a
                    residuals_a_i            = adjLogCounts_a_i - predLogCounts_a_i 

                    slopes_a[i]         = slope_a_i
                    intercepts_a[i]     = intercept_a_i
                    residuals_a[i]      = residuals_a_i

                if(update_consensus):
                    self.fitnesses.loc[variants, assay]       = slopes_a.ravel()
                    self.intercepts.loc[variants, assay]      = intercepts_a.ravel()
                    self.residuals.loc[variants, samples_a]   = residuals_a

                if(return_results):
                    slopes_all[:, a]     = slopes_a
                    intercepts_all[:, a] = intercepts_a
                    residuals_all[:, sample_col_counter:sample_col_counter+len(samples_a)] = residuals_a
                    sample_col_counter += len(samples_a)

            return (slopes_all, intercepts_all, residuals_all) if return_results else None


    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def infer_normalization_factors(self, assays=None, optimize_bias_prev=False, weighted_regression=False, bounds=None, method='BFGS', tol=1e-8, maxiter=None, verbose=False):  
        self.infer_sample_bias_prevalence(assays=assays, optimize_normalization_factors=True, optimize_bias_prev=optimize_bias_prev, weighted_regression=weighted_regression, bounds=bounds, method=method, tol=tol, maxiter=maxiter, verbose=verbose)
        return


    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def run(self, num_iters=5, optimize_bias_susc=True, optimize_bias_prev=True, optimize_normalization_factors=True, 
                                assays=None, variants=None, weighted_regression=True, bias_prev_bounds=None, bias_susc_bounds=None, 
                                method='BFGS', tol=1e-8, max_opt_iter=None, verbose=False, print_status=True):

        if(print_status):
            print(f"[ Stage 1: Inferring bias susceptibilities and bias prevalence deviations. ]")
        self.infer_bias_components(num_iters=num_iters, optimize_bias_susc=optimize_bias_susc, optimize_bias_prev=optimize_bias_prev, optimize_normalization_factors=optimize_normalization_factors, 
                                    assays=assays, variants=variants, weighted_regression=weighted_regression, bias_prev_bounds=bias_prev_bounds, bias_susc_bounds=bias_susc_bounds, 
                                    method=method, tol=tol, max_opt_iter=max_opt_iter, verbose=verbose)
    
        print(f"[ Stage 2: Inferring bias prevalence trends. ]                                    ")
        self.infer_bias_prevalence_trends(assays=assays)

        print(f"[ Computing bias-corrected fitness estimates. ]")
        self.perform_fits(assays=assays, min_mean_count=self.cfg['min_mean_count_linear_fits'], verbose=verbose)
        self.get_observed_counts(update_consensus=True)
        self.get_adjusted_counts(update_consensus=True)
        self.save(label='final', save_to_file=self.output_final)

        print(f"[ Done. ]")


    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def infer_bias_components(self, num_iters=5, optimize_bias_susc=True, optimize_bias_prev=True, optimize_normalization_factors=True, 
                                assays=None, variants=None, weighted_regression=True, bias_prev_bounds=None, bias_susc_bounds=None, 
                                method='BFGS', tol=1e-8, max_opt_iter=None, verbose=False):
    
        for i in range(1, num_iters+1):
    
            print(f"    Iteration #{i}                                                  ")
            
            self.infer_sample_bias_prevalence(assays=assays, optimize_bias_prev=optimize_bias_prev, optimize_normalization_factors=optimize_normalization_factors, weighted_regression=weighted_regression, bounds=bias_prev_bounds, method=method, tol=tol, maxiter=max_opt_iter, verbose=verbose)
            
            if(optimize_bias_susc):
                self.infer_variant_bias_susceptibility(variants=variants, weighted_regression=weighted_regression, method=method, bounds=bias_susc_bounds, tol=tol, maxiter=max_opt_iter, verbose=verbose)
            
            FORCE_STDEV_BIASPREV = self.cfg['target_bias_prevalence_stdev']
            biasScaleFactor = FORCE_STDEV_BIASPREV / np.std(self.samplesInfo['bias_prevalence'].values)
            self.samplesInfo['bias_prevalence'] *= biasScaleFactor
            self.variantsInfo['bias_susceptibility'] /= biasScaleFactor
            
            self.save(label=f"stage1-iter{i}", save_to_file=self.output_intermediates)


    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def infer_sample_bias_prevalence(self, assays=None, optimize_bias_prev=True, optimize_normalization_factors=True, weighted_regression=True, bounds=None, method='BFGS', tol=1e-8, maxiter=None, verbose=False):  # >>> NOTE: When using 'SLSQP' with tol=1e-8, initial normalization_factors optimization gets slightly different results than mikhail's code in some assays. Changing tol to 1e-10 under 'SLSQP' or changing the method to default (BFGS) with tol=1e-8 makes result align with mikhail's. Note that 'SLSQP' appear to be faster than default method even with tol=1e-10(2:3)

        if(not optimize_bias_prev and not optimize_normalization_factors):
            self._warning("Nothing to optimize, returning. (optimize_bias_prev==False and optimize_normalization_factors==False)")
            return

        assays = self.assays if assays is None else np.atleast_1d(assays)

        biassusc_allVariants = self.variantsInfo.loc[:, 'bias_susceptibility'].values

        for a, assay in enumerate(assays):
            print(f"        Inferring {'bias prevalences' if optimize_bias_prev else ''}{' and ' if optimize_bias_prev and optimize_normalization_factors else ''}{'normalization factors' if optimize_normalization_factors else ''}:", "Assay", assay, f"({a+1}/{len(assays)})", end='\n' if verbose else '                                \r')

            assayInfo  = self.samplesInfo.loc[(self.samplesInfo['assay'] == assay) & (~self.samplesInfo['timept'].isin(self.cfg['exclude_timepts']))].sort_values(by='timept')
            samples_a  = assayInfo['sample'].unique()
            biasprev_a = assayInfo['bias_prevalence'].values

            trustworthyVariants_a = np.where(self.trustworthy[assay]==True)[0]
            biassusc_a = biassusc_allVariants[trustworthyVariants_a]

            counts_a  = self.counts.loc[trustworthyVariants_a, samples_a].values
            counts_a[counts_a == 0] = self.cfg['pseudocount_value']
            weights_a = self.weights.loc[trustworthyVariants_a, samples_a].values if weighted_regression else np.ones_like(counts_a)

            normalization_factors_a = np.mean(counts_a, axis=0)

            #::::::::::::::::::::::::::::::::::::::::::::::::::
            def norm_residuals_over_variants(params_to_opt):
                biasprev_a_cur          = params_to_opt[:len(biasprev_a_opt)] if optimize_bias_prev else biasprev_a
                normalization_factors_a_cur_freevals = params_to_opt[-len(normalization_factors_a_opt_freevals):] 
                normalization_factors_a_cur          = np.concatenate([np.array([normalization_factors_a[0]]), normalization_factors_a_cur_freevals, np.array([normalization_factors_a[-1]])]) if optimize_normalization_factors else normalization_factors_a
                #-------------------------
                slopes_a, intercepts_a, residuals_a = self.perform_fits(counts=counts_a, weights=weights_a, normalization_factors=normalization_factors_a_cur, bias_prev=biasprev_a_cur, bias_susc=biassusc_a, return_results=True, update_consensus=False)
                residuals_a[~np.isfinite(residuals_a)] = 0.0 # convert any nan or inf residual values to 0 before calculating norm
                values_to_norm = np.concatenate([ (weights_a * residuals_a).ravel(order='F'), (self.cfg['sample_bias_penalty']*biasprev_a_cur), [((np.std(biasprev_a_cur)-self.cfg['target_bias_prevalence_stdev'])*self.cfg['sample_bias_penalty'])] ])
                return np.linalg.norm(values_to_norm)
            #::::::::::::::::::::::::::::::::::::::::::::::::::

            normalization_factors_a_opt          = np.copy(normalization_factors_a)
            normalization_factors_a_opt_freevals = np.copy(normalization_factors_a_opt[1:-1])
            biasprev_a_opt          = np.copy(biasprev_a)
            opt_results = scipy.optimize.minimize(norm_residuals_over_variants, x0=np.concatenate([biasprev_a_opt, normalization_factors_a_opt_freevals]), bounds=bounds, method=method, tol=tol, options=({'maxiter': maxiter} if maxiter is not None else {}))

            if(optimize_bias_prev):
                if(verbose):
                    print("OPTIMIZED BIASPREV:", opt_results['x'][:len(biasprev_a_opt)] )
                self.samplesInfo.loc[(self.samplesInfo['assay'] == assay) & (~self.samplesInfo['timept'].isin(self.cfg['exclude_timepts'])), 'bias_prevalence'] = opt_results['x'][:len(biasprev_a_opt)]
            if(optimize_normalization_factors):
                if(verbose):
                    print("OPTIMIZED BASELINE:", np.concatenate([np.array([normalization_factors_a[0]]), opt_results['x'][-len(normalization_factors_a_opt_freevals):], np.array([normalization_factors_a[-1]])]) )
                self.samplesInfo.loc[(self.samplesInfo['assay'] == assay) & (~self.samplesInfo['timept'].isin(self.cfg['exclude_timepts'])), 'normalization_factor'] = np.concatenate([np.array([normalization_factors_a[0]]), opt_results['x'][-len(normalization_factors_a_opt_freevals):], np.array([normalization_factors_a[-1]])]) 


    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def infer_variant_bias_susceptibility(self, variants=None, weighted_regression=True, method='BFGS', bounds=None, tol=1e-8, maxiter=None, verbose=False):  # >>> TODO: Possibly turn down these tolerances (from 1e-16) once it's clear that the optimization loop is working

        variants = self.variants if variants is None else np.atleast_1d(variants)

        trustworthyVariants_suscval = self.trustworthy.index[self.trustworthy['bias_susceptibility']==True].tolist()

        biassusc_allVariants = self.variantsInfo.loc[variants, 'bias_susceptibility'].values

        for i, variant in enumerate(variants):
            print(f"        Inferring bias susceptibilities:", f"Variant", f"({i+1}/{len(variants)})", end='\n' if verbose else '                                \r')

            if(variant not in trustworthyVariants_suscval):
                # This variant does not meet criteria for having a trustworthy, inferrable susceptibility value. Skip to next variant.
                continue

            trustworthyAssays_i  = [col for col in self.trustworthy.columns if 'bias' not in col]

            biassusc_i = biassusc_allVariants[i]

            counts_i_byAssay   = []
            weights_i_byAssay  = []
            normalization_factors_i_byAssay = []
            biasprev_i_byAssay = []
            numTimepts_byAssay = []
            for a, assay in enumerate(trustworthyAssays_i):
                assayInfo    = self.samplesInfo.loc[(self.samplesInfo['assay'] == assay) & (~self.samplesInfo['timept'].isin(self.cfg['exclude_timepts']))].sort_values(by='timept')
                samples_a    = assayInfo['sample'].unique()
                counts_a_i = self.counts.loc[i, samples_a].values
                counts_a_i[counts_a_i == 0] = self.cfg['pseudocount_value']
                counts_i_byAssay.append( counts_a_i )
                weights_i_byAssay.append( self.weights.loc[i, samples_a].values if weighted_regression else np.ones_like(counts_a_i) )
                normalization_factors_i_byAssay.append( assayInfo['normalization_factor'].values )
                biasprev_i_byAssay.append( assayInfo['bias_prevalence'].values )
                numTimepts_byAssay.append( len(samples_a) )
            #-------------------------
            allVariantAssaysSameNumTimepts = (len(set(numTimepts_byAssay)) == 1)
            if(allVariantAssaysSameNumTimepts):
                # All assays for this variant have same number of timepts
                counts_i_byAssay = np.array(counts_i_byAssay)
                weights_i_byAssay = np.array(weights_i_byAssay)
                normalization_factors_i_byAssay = np.array(normalization_factors_i_byAssay)
                biasprev_i_byAssay = np.array(biasprev_i_byAssay)
                numTimepts_byAssay = np.array(numTimepts_byAssay)

            #::::::::::::::::::::::::::::::::::::::::::::::::::
            def norm_residuals_over_assays(params_to_opt):
                biassusc_i_cur = params_to_opt[0]
                #-------------------------
                slopes_i, intercepts_i, residuals_i = self.perform_fits(counts=counts_i_byAssay, weights=weights_i_byAssay, normalization_factors=normalization_factors_i_byAssay, bias_prev=biasprev_i_byAssay, bias_susc=biassusc_i_cur, return_results=True, update_consensus=False, verbose=True)
                residuals_i[~np.isfinite(residuals_i)] = 0.0 # convert any nan or inf residual values to 0 before calculating norm
                if(allVariantAssaysSameNumTimepts):
                    weights_i = weights_i_byAssay
                else:
                    weights_i = np.zeros_like(residuals_i)
                    for a in range(weights_i.shape[0]):
                        weights_i[a, 0:len(weights_i_byAssay[a])] += weights_i_byAssay[a]

                values_to_norm = np.concatenate([ (weights_i * residuals_i).ravel(order='F'), [self.cfg['variant_bias_penalty']*biassusc_i_cur] ])
                return np.linalg.norm(values_to_norm)
            #::::::::::::::::::::::::::::::::::::::::::::::::::
                
            biassusc_i_opt = np.copy(biassusc_i)

            opt_results    = scipy.optimize.minimize(norm_residuals_over_assays, x0=biassusc_i_opt, bounds=bounds, method=method, tol=tol, options=({'maxiter': maxiter} if maxiter is not None else {}))

            if(verbose):
                print("OPTIMIZED BIASSUSC:", opt_results['x'][0] )
            
            self.variantsInfo.loc[variant, 'bias_susceptibility'] = opt_results['x'][0]


    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def infer_bias_prevalence_trends(self, assays=None):
    
        assays = self.assays if assays is None else np.atleast_1d(assays)
        
        biassusc_allVariants = self.variantsInfo.loc[:, 'bias_susceptibility'].values
        
        controlVariants = self.variantsInfo[self.variantsInfo['control_set'] == True].index.values

        for a, assay in enumerate(assays):
            
            assayInfo  = self.samplesInfo.loc[(self.samplesInfo['assay'] == assay) & (~self.samplesInfo['timept'].isin(self.cfg['exclude_timepts']))].sort_values(by='timept')
            samples_a  = assayInfo['sample'].unique()
            timepts_a  = assayInfo['timept'].values
            biasprev_a = assayInfo['bias_prevalence'].values
            
            trustworthyVariants_a = np.where(self.trustworthy[assay]==True)[0]
            trustworthyControlVariants_a = list(set(controlVariants).intersection(trustworthyVariants_a))
            
            biassusc_a = biassusc_allVariants[trustworthyControlVariants_a]
            
            fitnesses_controlSet_a  = self.fitnesses.loc[trustworthyControlVariants_a, assay].values
            
            #----
            
            delta_f_a = fitnesses_controlSet_a - fitnesses_controlSet_a.mean() # fitness misestimates among control set
            
            lambda_a, v_intcpt_a = self.linear_regression(x=biassusc_a, y=delta_f_a)     
            
            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~

            biasprev_a_abs = biasprev_a - lambda_a*timepts_a
            biasprev_a_abs = biasprev_a_abs - np.mean(biasprev_a_abs)

            self.samplesInfo.loc[(self.samplesInfo['assay'] == assay) & (~self.samplesInfo['timept'].isin(self.cfg['exclude_timepts'])), 'bias_prevalence'] = biasprev_a_abs


    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


##############################################
##############################################
##############################################
##############################################

    
if __name__ == '__main__':
    # Running as a script
    print("-- bias_correction.py --")

    import argparse
    parser  = argparse.ArgumentParser()
    parser.add_argument("-c", "--counts", required=True)
    parser.add_argument("-s", "--samples", required=True)
    parser.add_argument("-v", "--variants", required=True)
    parser.add_argument("-cfg", "--config", required=True)
    parser.add_argument("-o", "--outdir", required=True)
    parser.add_argument("--num_iters", default=5, type=int)
    parser.add_argument("--output_intermediates", default=True, type=bool)
    parser.add_argument("--output_final", default=True, type=bool)
    parser.add_argument("--seed", default=1)
    args    = parser.parse_args()

    np.random.seed(args.seed)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set up bias correction object:
    print("[ Initializing. ]")
    debiaser = BiasCorrection(counts=args.counts, 
                              samples=args.samples, 
                              variants=args.variants,
                              config=args.config,
                              outdir=args.outdir,
                              output_intermediates=args.output_intermediates,
                              output_final=args.output_final)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Perform bias correction:
    debiaser.run(num_iters=args.num_iters)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Save results to file
    # print("[ Saving outputs. ]")
    # debiaser.save(label='final', save_to_file=True)



           
            
















