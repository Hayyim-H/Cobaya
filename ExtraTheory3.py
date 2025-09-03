#####################################################################
from cobaya.theories.cosmo import BoltzmannBase
from classy import Class
import classy
import numpy as np

#####################################################################
from copy import deepcopy
from typing import NamedTuple, Sequence, Union, Optional, Callable, Any
from cobaya.tools import Pool1D, Pool2D, PoolND, combine_1d, get_compiled_import_path, \
    VersionCheckError
#####################################################################

class Collector(NamedTuple):
    method: str
    args: Sequence = []
    args_names: Sequence = []
    kwargs: dict = {}
    arg_array: Union[int, Sequence, None] = None
    z_pool: Optional[PoolND] = None
    post: Optional[Callable] = None

# default non linear code -- same as CAMB
non_linear_default_code = "hmcode"
non_linear_null_value = "none"

class ExtraTheory3(BoltzmannBase):
    
    def initialize(self):
        self.classy_massive = Class()
        self.classy_massless = Class()
        super().initialize()
        self.classy_module = classy
        self.derived_extra = []
        self.extra_args["output"] = self.extra_args.get("output", "")
        
        self.use_renames = True
    
        # Definimos el diccionario de traducciones directamente en el código.
        # Esto elimina la dependencia del fichero ExtraTheory3.yaml.
        self.renames = {
            # Background/geometry/Lambda
            'rdrag': 'rs_drag',
            'omegak': 'Omega_k',
            'omegal': 'Omega_Lambda',
            # Matter content
            'omegabh2': 'omega_b',
            'omegach2': 'omega_cdm',
            'Omega_m': 'Omega_m',
            # Inflation
            'As': 'A_s',
            'ns': 'n_s',
            # Thermodynamics
            'tau': 'tau_reio',
            'age': 'age'
            # Añade cualquier otro que necesites de tu lista original
        }
        
    def set_cl_reqs(self, reqs):
        if any(("t" in cl.lower()) for cl in reqs):
            self.extra_args["output"] += " tCl"
        if any((("e" in cl.lower()) or ("b" in cl.lower())) for cl in reqs):
            self.extra_args["output"] += " pCl"
        # For l_max_scalars, remember previous entries.
        self.extra_args["l_max_scalars"] = \
            max(self.extra_args.get("l_max_scalars", 0), max(reqs.values()))
        if 'T_cmb' not in self.derived_extra:
            self.derived_extra += ['T_cmb']
            
    def must_provide(self, **requirements):
        # Computed quantities required by the likelihood
        super().must_provide(**requirements)
        for k, v in self._must_provide.items():
            # Products and other computations
            if k == "Cl":
                self.set_cl_reqs(v)
                # For modern experiments, always lensed Cl's!
                self.extra_args["output"] += " lCl"
                self.extra_args["lensing"] = "yes"
                self.collectors[k] = Collector(
                    method="lensed_cl", kwargs={"lmax": self.extra_args["l_max_scalars"]})
            elif k == "unlensed_Cl":
                self.set_cl_reqs(v)
                self.collectors[k] = Collector(
                    method="raw_cl", kwargs={"lmax": self.extra_args["l_max_scalars"]})
            elif k == "Hubble":
                self.set_collector_with_z_pool(
                    k, v["z"], "Hubble", args_names=["z"], arg_array=0)
            elif k in ["Omega_b", "Omega_cdm", "Omega_nu_massive"]:
                func_name = {"Omega_b": "Om_b", "Omega_cdm": "Om_cdm",
                             "Omega_nu_massive": "Om_ncdm"}[k]
                self.set_collector_with_z_pool(
                    k, v["z"], func_name, args_names=["z"], arg_array=0)
            elif k == "angular_diameter_distance":
                self.set_collector_with_z_pool(
                    k, v["z"], "angular_distance", args_names=["z"], arg_array=0)
            elif k == "comoving_radial_distance":
                self.set_collector_with_z_pool(k, v["z"], "z_of_r", args_names=["z"],
                                               # returns r and dzdr!
                                               post=(lambda r, dzdr: r))
            elif k == "angular_diameter_distance_2":
                self.set_collector_with_z_pool(
                    k, v["z_pairs"], "angular_distance_from_to",
                    args_names=["z1", "z2"], arg_array=[0, 1], d=2)
            elif isinstance(k, tuple) and k[0] == "Pk_grid":
                self.extra_args["output"] += " mPk"
                v = deepcopy(v)
                self.add_P_k_max(v.pop("k_max"), units="1/Mpc")
                # NB: Actually, only the max z is used, and the actual sampling in z
                # for computing P(k,z) is controlled by `perturb_sampling_stepsize`
                # (default: 0.1). But let's leave it like this in case this changes
                # in the future.
                self.add_z_for_matter_power(v.pop("z"))
                if v["nonlinear"]:
                    if "non_linear" not in self.extra_args:
                        # this is redundant with initialisation, but just in case
                        self.extra_args["non_linear"] = non_linear_default_code
                    elif self.extra_args["non_linear"] == non_linear_null_value:
                        raise LoggedError(
                            self.log, ("Non-linear Pk requested, but `non_linear: "
                                       f"{non_linear_null_value}` imposed in "
                                       "`extra_args`"))
                pair = k[2:]
                if pair == ("delta_tot", "delta_tot"):
                    v["only_clustering_species"] = False
                    self.collectors[k] = Collector(
                        method="get_pk_and_k_and_z",
                        kwargs=v)
                elif pair == ("delta_nonu", "delta_nonu"):
                    v["only_clustering_species"] = True
                    self.collectors[k] = Collector(
                        method="get_pk_and_k_and_z", kwargs=v)
                elif pair == ("Weyl", "Weyl"):
                    self.extra_args["output"] += " mTk"
                    self.collectors[k] = Collector(
                        method="get_Weyl_pk_and_k_and_z", kwargs=v)
                else:
                    raise LoggedError(self.log, "NotImplemented in CLASS: %r", pair)
            elif k == "sigma8_z":
                self.add_z_for_matter_power(v["z"])
                self.set_collector_with_z_pool(
                    k, v["z"], "sigma", args=[8], args_names=["R", "z"],
                    kwargs={"h_units": True}, arg_array=1)
            elif k == "fsigma8":
                self.add_z_for_matter_power(v["z"])
                z_step = 0.1  # left to CLASS default; increasing does not appear to help
                self.set_collector_with_z_pool(
                    k, v["z"], "effective_f_sigma8", args=[z_step],
                    args_names=["z", "z_step"], arg_array=0)
            elif isinstance(k, tuple) and k[0] == "sigma_R":
                self.extra_args["output"] += " mPk"
                self.add_P_k_max(v.pop("k_max"), units="1/Mpc")
                # NB: See note about redshifts in Pk_grid
                self.add_z_for_matter_power(v["z"])
                pair = k[1:]
                try:
                    method = {("delta_tot", "delta_tot"): "sigma",
                              ("delta_nonu", "delta_nonu"): "sigma_cb"}[pair]
                except KeyError as excpt:
                    raise LoggedError(
                        self.log, f"sigma(R,z) not implemented for {pair}") from excpt
                self.collectors[k] = Collector(
                    method=method, kwargs={"h_units": False}, args=[v["R"], v["z"]],
                    args_names=["R", "z"], arg_array=[[0], [1]],
                    post=(lambda R, z, sigma: (z, R, sigma.T)))
            elif k in [f"CLASS_{q}" for q in ["background", "thermodynamics",
                                              "primordial", "perturbations", "sources"]]:
                # Get direct CLASS results
                self.collectors[k] = Collector(method=f"get_{k.lower()[len('CLASS_'):]}")
            elif v is None:
                k_translated = self.translate_param(k)
                if k_translated not in self.derived_extra:
                    self.derived_extra += [k_translated]
            else:
                raise LoggedError(self.log, "Requested product not known: %r", {k: v})
        # Derived parameters (if some need some additional computations)
        if any(("sigma8" in s) for s in set(self.output_params).union(requirements)):
            self.extra_args["output"] += " mPk"
            self.add_P_k_max(1, units="1/Mpc")
        # Adding tensor modes if requested
        if self.extra_args.get("r") or "r" in self.input_params:
            self.extra_args["modes"] = "s,t"
        # If B spectrum with l>50, or lensing, recommend using a non-linear code
        cls = self._must_provide.get("Cl", {})
        has_BB_l_gt_50 = (any(("b" in cl.lower()) for cl in cls) and
                          max(cls[cl] for cl in cls if "b" in cl.lower()) > 50)
        has_lensing = any(("p" in cl.lower()) for cl in cls)
        if (has_BB_l_gt_50 or has_lensing) and \
                self.extra_args.get("non_linear") == non_linear_null_value:
            self.log.warning("Requesting BB for ell>50 or lensing Cl's: "
                             "using a non-linear code is recommended (and you are not "
                             "using any). To activate it, set "
                             "'non_linear: halofit|hmcode|...' in classy's 'extra_args'.")
        # Cleanup of products string
        self.extra_args["output"] = " ".join(set(self.extra_args["output"].split()))
        self.check_no_repeated_input_extra()
        
    def add_z_for_matter_power(self, z):
        if getattr(self, "z_for_matter_power", None) is None:
            self.z_for_matter_power = np.empty(0)
        self.z_for_matter_power = np.flip(combine_1d(z, self.z_for_matter_power))
        self.extra_args["z_pk"] = " ".join(["%g" % zi for zi in self.z_for_matter_power])
        
    def set_collector_with_z_pool(self, k, zs, method, args=(), args_names=(),
                                  kwargs=None, arg_array=None, post=None, d=1):
        """
        Creates a collector for a z-dependent quantity, keeping track of the pool of z's.

        If ``z`` is an arg, i.e. it is in ``args_names``, then omit it in the ``args``,
        e.g. ``args_names=["a", "z", "b"]`` should be passed together with
        ``args=[a_value, b_value]``.
        """
        if k in self.collectors:
            z_pool = self.collectors[k].z_pool
            z_pool.update(zs)
        else:
            Pool = {1: Pool1D, 2: Pool2D}[d]
            z_pool = Pool(zs)
        # Insert z as arg or kwarg
        kwargs = kwargs or {}
        if d == 1 and "z" in kwargs:
            kwargs = deepcopy(kwargs)
            kwargs["z"] = z_pool.values
        elif d == 1 and "z" in args_names:
            args = deepcopy(args)
            i_z = args_names.index("z")
            args = list(args[:i_z]) + [z_pool.values] + list(args[i_z:])
        elif d == 2 and "z1" in args_names and "z2" in args_names:
            # z1 assumed appearing before z2!
            args = deepcopy(args)
            i_z1 = args_names.index("z1")
            i_z2 = args_names.index("z2")
            args = (list(args[:i_z1]) + [z_pool.values[:, 0]] + list(args[i_z1:i_z2]) +
                    [z_pool.values[:, 1]] + list(args[i_z2:]))
        else:
            raise LoggedError(
                self.log,
                f"I do not know how to insert the redshift for collector method {method} "
                f"of requisite {k}")
        self.collectors[k] = Collector(
            method=method, z_pool=z_pool, args=args, args_names=args_names, kwargs=kwargs,
            arg_array=arg_array, post=post)
            
    def add_P_k_max(self, k_max, units):
        r"""
        Unifies treatment of :math:`k_\mathrm{max}` for matter power spectrum:
        ``P_k_max_[1|h]/Mpc``.

        Make ``units="1/Mpc"|"h/Mpc"``.
        """
        # Fiducial h conversion (high, though it may slow the computations)
        h_fid = 1
        if units == "h/Mpc":
            k_max *= h_fid
        # Take into account possible manual set of P_k_max_***h/Mpc*** through extra_args
        k_max_old = self.extra_args.pop(
            "P_k_max_1/Mpc", h_fid * self.extra_args.pop("P_k_max_h/Mpc", 0))
        self.extra_args["P_k_max_1/Mpc"] = max(k_max, k_max_old)

    def set(self, params_values_dict):
        # If no output requested, remove arguments that produce an error
        self.classy_massive.empty()
        self.classy_massless.empty()
        if not self.extra_args["output"]:
            for k in ["non_linear"]:
                self.extra_args.pop(k, None)
        # Prepare parameters to be passed: this-iteration + extra
        args = {self.translate_param(p): v for p, v in params_values_dict.items()}
        args.update(self.extra_args)
        # For Hayyim's purposes:
        self.sum_eff = params_values_dict.get('sum_mnu_eff')
        self.num_neutrinos = self.extra_args.get('N_ncdm')
        args_massless, args_massive = args.copy(), args.copy()
        if self.num_neutrinos==1:
            args_massless['m_ncdm'] = 0
            args_massive['m_ncdm'] = abs(self.sum_eff)
        elif self.num_neutrinos==3:
            args_massless['m_ncdm'] = '0,0,0'
            individual_mass = abs(self.sum_eff) / self.num_neutrinos
            args_massive['m_ncdm'] = f"{individual_mass},{individual_mass},{individual_mass}"
        args_massless.pop('sum_mnu_eff')
        args_massive.pop('sum_mnu_eff')
        # Generate and save
        self.log.debug("Setting parameters: %r", args_massive)
        
        self.classy_massive.set(**args_massive)
        self.classy_massless.set(**args_massless) 
        
#######################################################################################

    def calculate(self, state: dict, want_derived: bool = True, **params_values_dict):
        self.set(params_values_dict)
        
        try:
            if self.sum_eff >= 0:
                self.classy_massive.compute()
            else:
                self.classy_massive.compute()
                self.classy_massless.compute()
        except self.classy_module.CosmoComputationError as e:
            self.log.debug(f"CLASS computation failed. Error: {e}")
            return False
        
        for product, collector in self.collectors.items():
            state[product] = self._get_combined_product(collector)

        d, d_extra = self._get_derived_all(derived_requested=want_derived)
        if want_derived:
            state["derived"] = {p: d.get(self.translate_param(p)) for p in self.output_params}
        state["derived_extra"] = d_extra
        return True

#######################################################################################

    def _get_combined_product(self, collector):
        method_name = collector.method
        if self.sum_eff >= 0:
            method = getattr(self.classy_massive, method_name)
            return method(*collector.args, **collector.kwargs)
        
        # Lógica para masa < 0:
        method_massless = getattr(self.classy_massless, method_name)
        res_massless = method_massless(*collector.args, **collector.kwargs)
        method_massive = getattr(self.classy_massive, method_name)
        res_massive = method_massive(*collector.args, **collector.kwargs)

        # Caso para espectros del CMB (diccionarios) - Generalmente no necesita interpolación
        if isinstance(res_massless, dict):
            combined_res = {}
            for k, v_massless in res_massless.items():
                if k == 'ell':
                    combined_res[k] = v_massless
                else:
                    v_massive = res_massive.get(k, 0)
                    # Asumimos que los 'ell' son siempre los mismos
                    combined_res[k] = 2 * v_massless - v_massive
            return combined_res
            
        # Caso para espectro de materia (tuplas) - AQUÍ APLICAMOS LA INTERPOLACIÓN
        elif isinstance(res_massless, (list, tuple)):
            # Desempaquetamos. Las redes de 'k' pueden ser diferentes.
            P_massless, k_massless, z_vals = res_massless
            P_massive, k_massive, _ = res_massive

            # Convertimos a arrays de NumPy para operar
            P_massless = np.array(P_massless)
            P_massive = np.array(P_massive)
            k_massless = np.array(k_massless)
            k_massive = np.array(k_massive)

            # Verificamos si las redes de k son idénticas. Si lo son, restamos directamente.
            if P_massless.shape == P_massive.shape and np.allclose(k_massless, k_massive):
                pk_extrapolated_raw = 2 * P_massless - P_massive
            else:
                # --- LÓGICA DE INTERPOLACIÓN ---
                # Si las redes son distintas, interpolamos P_massive a la red de k_massless
                self.log.debug("Different k-grids found for P(k). Interpolating...")
                
                # Preparamos un array vacío para el resultado interpolado
                P_massive_interp = np.zeros_like(P_massless)
                
                # Interpolamos cada rebanada de redshift. Es más preciso interpolar en el espacio logarítmico.
                for i in range(P_massless.shape[1]): # Iteramos sobre los redshifts
                    log_p_massive_slice = np.log(P_massive[:, i])
                    # Usamos np.interp para evaluar P_massive en los puntos de k_massless
                    log_p_interp_slice = np.interp(np.log(k_massless), np.log(k_massive), log_p_massive_slice)
                    P_massive_interp[:, i] = np.exp(log_p_interp_slice)
                
                # Ahora realizamos la extrapolación con arrays de formas compatibles
                pk_extrapolated_raw = 2 * P_massless - P_massive_interp

            # Transponemos el resultado para que tenga la forma [redshift, k]
            pk_final = pk_extrapolated_raw.T
            
            # Nos aseguramos de que sea 2D
            if pk_final.ndim == 1:
                pk_final = np.atleast_2d(pk_final)

            # Devolvemos la tupla en el orden estándar (k, z, Pk)
            return (k_massless, z_vals, pk_final)
            
        # Caso para escalares
        elif isinstance(res_massless, (np.ndarray, np.isscalar)):
            return 2 * res_massless - res_massive
        
        # Fallback de seguridad
        return res_massless

#######################################################################################
    # --- NUEVOS MÉTODOS PARA SOBRESCRIBIR LA CLASE BASE ---
    
    def get_Cl(self, ell_factor=False, units="FIRASmuK2"):
        """
        Devuelve los espectros de potencias CON LENTES.
        Este método sobrescribe el de la clase base para evitar el doble procesamiento.
        """
        cls = deepcopy(self.current_state["Cl"])

        if not ell_factor:
            return cls

        # Aplica el factor l(l+1)/2pi si se solicita
        ells = cls["ell"]
        ells_factor = (ells * (ells + 1) / (2 * np.pi))
        
        for cl_key in cls:
            if cl_key == "ell": continue
            
            # Aplica el factor correspondiente a cada tipo de espectro
            if "p" not in cl_key: # TT, EE, BB, TE
                cls[cl_key] *= ells_factor
            elif cl_key == "pp": # Lensing potential
                cls[cl_key] *= ells_factor**2 * (2 * np.pi)
            else: # Correlaciones cruzadas con lensing (TP, EP)
                cls[cl_key] *= ells_factor**(3/2) * np.sqrt(2 * np.pi)
        return cls

    def get_unlensed_Cl(self, ell_factor=False, units="FIRASmuK2"):
        """
        Devuelve los espectros de potencias SIN LENTES.
        Este método sobrescribe el de la clase base.
        """
        # CLASS devuelve los Cl sin lentes en unidades K^2. Necesitamos el factor de conversión.
        # La forma más segura es tomar T_cmb de los parámetros derivados ya calculados.
        T_cmb = self.current_state['derived_extra']['T_cmb']
        factor = self._cmb_unit_factor(units, T_cmb)

        cls = deepcopy(self.current_state["unlensed_Cl"])
        
        for cl_key in cls:
            if cl_key == "ell": continue
            # Aplicar conversión de unidades y, opcionalmente, ell_factor
            cls[cl_key] *= factor**2
            if ell_factor:
                cls[cl_key] *= (cls["ell"] * (cls["ell"] + 1) / (2 * np.pi))
        return cls



##########################################################################################
            
    # (Pega esto dentro de tu clase ExtraTheory6, reemplazando la función existente)

    def _get_derived_all(self, derived_requested=True):

        requested_class_names = [self.translate_param(p) for p in self.output_params] if derived_requested else []
        all_requested_class_names = set(requested_class_names).union(self.derived_extra)

        if not all_requested_class_names:
            return {}, {}

        # Si sum_eff > 0, solo necesitamos el cálculo masivo.
        if self.sum_eff > 0:
            combined_derived = self._get_class_derived_params(self.classy_massive, all_requested_class_names)
        else: # Esto cubre sum_eff <= 0
            # Siempre necesitamos el cálculo sin masa.
            derived_params_massless = self._get_class_derived_params(self.classy_massless, all_requested_class_names)

            if self.sum_eff == 0:
                # Si la masa es cero, el resultado es el del caso sin masa.
                combined_derived = derived_params_massless
            else: 
                # Para masa negativa, necesitamos ambos para la extrapolación.
                derived_params_massive = self._get_class_derived_params(self.classy_massive, all_requested_class_names)
                
                combined_derived = {}
                for p_name in all_requested_class_names:
                    val_massless = derived_params_massless.get(p_name)
                    val_massive = derived_params_massive.get(p_name)
                    
                    # Aplicar la fórmula solo a valores numéricos
                    if isinstance(val_massless, (int, float, np.number)) and isinstance(val_massive, (int, float, np.number)):
                        combined_derived[p_name] = 2*val_massless - val_massive
        
        # Separar los resultados en los diccionarios de salida
        derived = {p: combined_derived.get(self.translate_param(p)) for p in self.output_params} if derived_requested else {}
        derived_extra = {p: combined_derived.get(p) for p in self.derived_extra}
        
        return derived, derived_extra

    def _get_class_derived_params(self, classy_instance, requested_params_set):
        """Función auxiliar para obtener los parámetros derivados de una instancia de CLASS."""
        derived_params = {}
        
        # Manejar parámetros con métodos de acceso especiales
        if "rs_drag" in requested_params_set:
            derived_params["rs_drag"] = classy_instance.rs_drag()
        if "Omega_nu" in requested_params_set:
            derived_params["Omega_nu"] = classy_instance.Omega_nu
        if "T_cmb" in requested_params_set:
            derived_params["T_cmb"] = classy_instance.T_cmb()

        # Obtener el resto con el método general
        general_params_to_get = [p for p in requested_params_set if p not in derived_params]
        if general_params_to_get:
            derived_params.update(
                classy_instance.get_current_derived_parameters(general_params_to_get))
                
        return derived_params     
 
#########################################################################################

    def get_CLASS_background(self):
        """Direct access to ``get_background`` from the CLASS python interface."""
        return self.current_state["CLASS_background"]

    def get_CLASS_thermodynamics(self):
        """Direct access to ``get_thermodynamics`` from the CLASS python interface."""
        return self.current_state["CLASS_thermodynamics"]

    def get_CLASS_primordial(self):
        """Direct access to ``get_primordial`` from the CLASS python interface."""
        return self.current_state["CLASS_primordial"]

    def get_CLASS_perturbations(self):
        """Direct access to ``get_perturbations`` from the CLASS python interface."""
        return self.current_state["CLASS_perturbations"]

    def get_CLASS_sources(self):
        """Direct access to ``get_sources`` from the CLASS python interface."""
        return self.current_state["CLASS_sources"]
        
    def get_can_provide_params(self):
        names = ["h", "H0", "Omega_Lambda", "Omega_cdm", "Omega_b", "Omega_m", "Omega_k",
                 "rs_drag", "tau_reio", "z_reio", "z_rec", "tau_rec", "m_ncdm_tot",
                 "Neff", "YHe", "age", "conformal_age", "sigma8", "sigma8_cb",
                 "theta_s_100"]
        for name, mapped in self.renames.items():
            if mapped in names:
                names.append(name)
        return names
        
    def close(self):
        self.classy_massive.empty()
        self.classy_massless.empty()
