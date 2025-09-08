# Global
import sys
import os
import platform
from copy import deepcopy
from typing import NamedTuple, Sequence, Union, Optional, Callable, Any
import numpy as np
import classy
from classy import Class

# Local
from cobaya.theories.cosmo import BoltzmannBase
from cobaya.log import LoggedError, get_logger
from cobaya.install import download_github_release, pip_install, check_gcc_version
from cobaya.component import ComponentNotInstalledError, load_external_module
from cobaya.tools import Pool1D, Pool2D, PoolND, combine_1d, get_compiled_import_path, \
    VersionCheckError


# Result collector
# NB: cannot use kwargs for the args, because the CLASS Python interface
#     is C-based, so args without default values are not named.
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
        # Add general CLASS stuff
        self.extra_args["output"] = self.extra_args.get("output", "")
        if "sBBN file" in self.extra_args:
            self.extra_args["sBBN file"] = (
                self.extra_args["sBBN file"].format(classy=self.path))
        # Normalize `non_linear` vs `non linear`: prefer underscore
        # Keep this convention throughout the rest of this module!
        if "non linear" in self.extra_args:
            if "non_linear" in self.extra_args:
                raise LoggedError(
                    self.log, ("In `extra_args`, only one of `non_linear` or `non linear`"
                               " should be defined."))
            self.extra_args["non_linear"] = self.extra_args.pop("non linear")
        # Normalize non_linear None|False --> "none"
        # Use default one if not specified
        if self.extra_args.get("non_linear", "dummy_string") in (None, False):
            self.extra_args["non_linear"] = non_linear_null_value
        elif ("non_linear" not in self.extra_args or
              self.extra_args["non_linear"] is True):
            self.extra_args["non_linear"] = non_linear_default_code
        # Derived parameters that may not have been requested, but will be necessary later
        self.derived_extra = []
        
        self.use_renames = True
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
        """
        Sets some common settings for both lensend and unlensed Cl's.
        """
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
                        kwargs=v,
                        post=(lambda P, kk, z: (kk, z, np.array(P).T)))
                elif pair == ("delta_nonu", "delta_nonu"):
                    v["only_clustering_species"] = True
                    self.collectors[k] = Collector(
                        method="get_pk_and_k_and_z", kwargs=v,
                        post=(lambda P, kk, z: (kk, z, np.array(P).T)))
                elif pair == ("Weyl", "Weyl"):
                    self.extra_args["output"] += " mTk"
                    self.collectors[k] = Collector(
                        method="get_Weyl_pk_and_k_and_z", kwargs=v,
                        post=(lambda P, kk, z: (kk, z, np.array(P).T)))
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
        elif self.num_neutrinos!=1:
            args_massless['m_ncdm'] = '0,0,0'
            individual_mass = abs(self.sum_eff) / self.num_neutrinos
            args_massive['m_ncdm'] = f"{individual_mass},{individual_mass},{individual_mass}"
        args_massless.pop('sum_mnu_eff')
        args_massive.pop('sum_mnu_eff')
        # Generate and save
        self.log.debug("Setting parameters: %r", args)
        
        self.classy_massive.set(**args_massive)
        self.classy_massless.set(**args_massless) 
        
#########################################################################################
    def _get_observable(self, method_massless, method_massive, *args, **kwargs):
        if self.sum_eff < 0:
            
            # Realizamos los DOS cálculos
            result_massless = method_massless(*args, **kwargs)
            result_massive = method_massive(*args, **kwargs)

            # Caso 1: El resultado es una TUPLA (p. ej., de Pk_grid)
            if isinstance(result_massive, tuple):
                # Asumimos que las coordenadas (todo menos el último elemento) son iguales
                coords = result_massive[:-1]
                # La fórmula se aplica solo al último elemento (el array de valores)
                val_massless = result_massless[-1]
                val_massive = result_massive[-1]
                extrapolated_val = val_massless - (val_massive - val_massless)
                # Reconstruimos la tupla final
                return coords + (extrapolated_val,)

            # Caso 2: El resultado es un DICCIONARIO (p. ej., de Cl's)
            elif isinstance(result_massive, dict):
                extrapolated_dict = {}
                for key, val_massive in result_massive.items():
                    # La clave 'ell' solo se copia, no se extrapola
                    if key == 'ell':
                        extrapolated_dict[key] = val_massive
                        continue
                    val_massless = result_massless[key]
                    # Aplicamos la fórmula a cada array del diccionario
                    extrapolated_dict[key] = val_massless - (val_massive - val_massless)
                return extrapolated_dict

            # Caso 3: El resultado es un NÚMERO (float, int)
            else:
                return 2*result_massless - result_massive

        else:
            # Si la masa es positiva, el cálculo es el estándar.
            return method_massive(*args, **kwargs)
#########################################################################################

    def calculate(self, state, want_derived=True, **params_values_dict):
        # Set parameters
        self.set(params_values_dict)
        # Compute!
        try:
            self.classy_massive.compute()
            if self.sum_eff<0:
                self.classy_massless.compute()
        # "Valid" failure of CLASS: parameters too extreme -> log and report
        except classy.CosmoComputationError as e:
            if self.stop_at_error:
                self.log.error(
                    "Computation error (see traceback below)! "
                    "Parameters sent to CLASS: %r and %r.\n"
                    "To ignore this kind of error, make 'stop_at_error: False'.",
                    state["params"], dict(self.extra_args))
                raise
            else:
                self.log.debug("Computation of cosmological products failed. "
                               "Assigning 0 likelihood and going on. "
                               "The output of the CLASS error was %s", e)
            return False
        # CLASS not correctly initialized, or input parameters not correct
        except classy.CosmoSevereError:
            self.log.error("Serious error setting parameters or computing results. "
                           "The parameters passed were %r and %r. To see the original "
                           "CLASS' error traceback, make 'debug: True'.",
                           state["params"], self.extra_args)
            raise  # No LoggedError, so that CLASS traceback gets printed
        # Gather products
        for product, collector in self.collectors.items():
            # Special case: sigma8 needs H0, which cannot be known beforehand:
            if "sigma8" in self.collectors:
                self.collectors["sigma8"].args[0] = 8 / self.classy_massive.h()
            method_massive = getattr(self.classy_massive, collector.method)
            method_massless = getattr(self.classy_massless, collector.method)
            arg_array = self.collectors[product].arg_array
            if isinstance(arg_array, int):
                arg_array = np.atleast_1d(arg_array)
            if arg_array is None:
                state[product] = self._get_observable(method_massless, method_massive, *self.collectors[product].args, **self.collectors[product].kwargs)

            elif isinstance(arg_array, Sequence) or isinstance(arg_array, np.ndarray):
                arg_array = np.array(arg_array)
                if len(arg_array.shape) == 1:
                    # if more than one vectorised arg, assume all vectorised in parallel
                    n_values = len(self.collectors[product].args[arg_array[0]])
                    state[product] = np.zeros(n_values)
                    args = deepcopy(list(self.collectors[product].args))
                    for i in range(n_values):
                        for arg_arr_index in arg_array:
                            args[arg_arr_index] = \
                                self.collectors[product].args[arg_arr_index][i]
                        state[product][i] = self._get_observable(method_massless, method_massive,
                            *args, **self.collectors[product].kwargs)
                elif len(arg_array.shape) == 2:
                    if len(arg_array) > 2:
                        raise NotImplementedError("Only 2 array expanded vars so far.")
                    # Create outer combinations
                    x_and_y = np.array(np.meshgrid(
                        self.collectors[product].args[arg_array[0, 0]],
                        self.collectors[product].args[arg_array[1, 0]])).T
                    args = deepcopy(list(self.collectors[product].args))
                    result = np.empty(shape=x_and_y.shape[:2])
                    for i, row in enumerate(x_and_y):
                        for j, column_element in enumerate(x_and_y[i]):
                            args[arg_array[0, 0]] = column_element[0]
                            args[arg_array[1, 0]] = column_element[1]
                            result[i, j] = self._get_observable(method_massless, method_massive,
                                *args, **self.collectors[product].kwargs)
                    state[product] = (
                        self.collectors[product].args[arg_array[0, 0]],
                        self.collectors[product].args[arg_array[1, 0]], result)
                else:
                    raise ValueError("arg_array not correctly formatted.")
            elif arg_array in self.collectors[product].kwargs:
                value = np.atleast_1d(self.collectors[product].kwargs[arg_array])
                state[product] = np.zeros(value.shape)
                for i, v in enumerate(value):
                    kwargs = deepcopy(self.collectors[product].kwargs)
                    kwargs[arg_array] = v
                    state[product][i] = self._get_observable(method_massless, method_massive,
                        *self.collectors[product].args, **kwargs)
            else:
                raise LoggedError(self.log, "Variable over which to do an array call "
                                            f"not known: arg_array={arg_array}")
            if collector.post:
                state[product] = collector.post(*state[product])
        # Prepare derived parameters
        d, d_extra = self._get_derived_all(derived_requested=want_derived)
        if want_derived:
            state["derived"] = {p: d.get(p) for p in self.output_params}
            # Prepare necessary extra derived parameters
        state["derived_extra"] = deepcopy(d_extra)

    def _get_derived_all(self, derived_requested=True):
        requested = [self.translate_param(p) for p in (
            self.output_params if derived_requested else [])]
        requested_and_extra = dict.fromkeys(set(requested).union(self.derived_extra))
        # Parameters with their own getters or different CLASS internal names
        if "rs_drag" in requested_and_extra:
            if self.sum_eff<0:
                requested_and_extra["rs_drag"] = 2*self.classy_massless.rs_drag() -self.classy_massive.rs_drag()
            else:
                requested_and_extra["rs_drag"] = self.classy_massive.rs_drag()
        if "Omega_nu" in requested_and_extra:
            if self.sum_eff<0:
                requested_and_extra["Omega_nu"] = 2*self.classy_massless.Omega_nu -self.classy_massive.Omega_nu
            else:
                requested_and_extra["Omega_nu"] = self.classy_massive.Omega_nu
        if "T_cmb" in requested_and_extra:
            if self.sum_eff<0:
                requested_and_extra["T_cmb"] = 2*self.classy_massless.T_cmb() -self.classy_massive.T_cmb()
            else:
                requested_and_extra["T_cmb"] = self.classy_massive.T_cmb()
        # Primero, obtenemos la lista de parámetros que necesitamos calcular
        params_to_get = [p for p, v in requested_and_extra.items() if v is None]

        # Comprobamos si la masa es negativa para decidir qué hacer
        if self.sum_eff < 0:
            
            # 1. Obtenemos los dos diccionarios de resultados
            derived_params_massless = self.classy_massless.get_current_derived_parameters(params_to_get)
            derived_params_massive = self.classy_massive.get_current_derived_parameters(params_to_get)
            
            # 2. Iteramos sobre la lista de parámetros para aplicar la fórmula a cada uno
            for p_name in params_to_get:
                val_massless = derived_params_massless[p_name]
                val_massive = derived_params_massive[p_name]
                
                # Aplicamos la Ec. 7 al par de valores y actualizamos el diccionario principal
                requested_and_extra[p_name] = 2 * val_massless - val_massive
                
        else:
            # Si la masa es positiva, el caso es simple: solo un cálculo
            derived_params_massive = self.classy_massive.get_current_derived_parameters(params_to_get)
            requested_and_extra.update(derived_params_massive)

        # El resto del código no necesita cambios
        # Separate the parameters before returning
        derived = {
            p: requested_and_extra[self.translate_param(p)] for p in self.output_params}
        derived_extra = {p: requested_and_extra[p] for p in self.derived_extra}
        return derived, derived_extra

    def _get_Cl(self, ell_factor=False, units="FIRASmuK2", lensed=True):
        which_key = "Cl" if lensed else "unlensed_Cl"
        which_error = "lensed" if lensed else "unlensed"
        try:
            cls = deepcopy(self.current_state[which_key])
        except Exception as excpt:
            raise LoggedError(
                self.log,
                "No %s Cl's were computed. Are you sure that you have requested them?",
                which_error
            ) from excpt
        # unit conversion and ell_factor
        ells_factor = \
            ((cls["ell"] + 1) * cls["ell"] / (2 * np.pi))[2:] if ell_factor else 1
        units_factor = self._cmb_unit_factor(
            units, self.current_state['derived_extra']['T_cmb'])
        for cl in cls:
            if cl == "ell":
                continue
            units_power = float(sum(cl.count(p) for p in ["t", "e", "b"]))
            cls[cl][2:] *= units_factor ** units_power
            if ell_factor:
                if "p" not in cl:
                    cls[cl][2:] *= ells_factor
                elif cl == "pp" and lensed:
                    cls[cl][2:] *= ells_factor ** 2 * (2 * np.pi)
                elif "p" in cl and lensed:
                    cls[cl][2:] *= ells_factor ** (3 / 2) * np.sqrt(2 * np.pi)
        return cls

    def get_Cl(self, ell_factor=False, units="FIRASmuK2"):
        return self._get_Cl(ell_factor=ell_factor, units=units, lensed=True)

    def get_unlensed_Cl(self, ell_factor=False, units="FIRASmuK2"):
        return self._get_Cl(ell_factor=ell_factor, units=units, lensed=False)

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

    def close(self):
        self.classy_massive.empty()
        self.classy_massless.empty()

    def get_can_provide_params(self):
        names = ["h", "H0", "Omega_Lambda", "Omega_cdm", "Omega_b", "Omega_m", "Omega_k",
                 "rs_drag", "tau_reio", "z_reio", "z_rec", "tau_rec", "m_ncdm_tot",
                 "Neff", "YHe", "age", "conformal_age", "sigma8", "sigma8_cb",
                 "theta_s_100"]
        for name, mapped in self.renames.items():
            if mapped in names:
                names.append(name)
        return names

    def get_can_support_params(self):
        # non-exhaustive list of supported input parameters that will be assigned to
        # classy if they are varied
        return ['H0']
