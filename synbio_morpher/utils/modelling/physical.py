
# Copyright (c) 2023, Olivia Gallup
# All rights reserved.

# This source code is licensed under the MIT-style license found in the
# LICENSE file in the root directory of this source tree. 
    
import numpy as jnp
import jax.numpy as jnp
from synbio_morpher.utils.misc.units import SCIENTIFIC


# def F(E):
#     """ Parameterisation of relative GFP fluorescence binding from 
#     paper [Metabolic engineering of Escherichia coli using synthetic small regulatory RNAs](
#     https://www.nature.com/articles/nbt.2461
#     ). See the notebook `explanations/binding_energy_reparameterisation` for 
#     more details.
#     The binding energy is in units of kcal/mol """
#     F = (1-0.01)/(1+jnp.exp(-(E/2 + 5))) + 0.01
#     return F


def equilibrium_constant_reparameterisation(E, initial: jnp.ndarray):
    """ Input: E is $\Delta G$ of binding in kcal/mol. 
    Output: equilibrium constant

    IMPORTANT: Using the mean initial quantity of all species, as this 
    equation was derived under the assumption that all unbound species 
    start with the same concentration and have the same interactions """
    # return 1/initial * (1/F(E) - 1)
    Fs = jnp.exp(-0.8 * (E + 10))
    return Fs/initial


def gibbs_K(E):
    """ In J/mol. dG = - RT ln(K) """
    K = jnp.exp(jnp.divide(-E, SCIENTIFIC['RT']))
    return K


def gibbs_K_cal(E):
    """ Translate interaction binding energy (kcal) to the
    equilibrium rate of binding.
    AG = - RT ln(K)
    AG = - RT ln(kb/kd)
    K = e^(- G / RT)
    """
    K = jnp.exp(jnp.divide(-E, SCIENTIFIC['RT_cal']))
    return K


def eqconstant_to_rates(eqconstants, k_f):
    """ Translate the equilibrium rate of binding to
    the rate of binding (either association or dissociation
    rate - in this case dissociation). Input in mol, output in molecules:
    k_f: binding rate per Ms
    eqconstants: unitless but in terms of mol
    k_r: unbinding rate per s"""
    
    k_r = jnp.divide(k_f, eqconstants)
    return k_f*jnp.ones_like(k_r), k_r


def rates_to_eqconstant(k_f, k_r):
    eqconstants = jnp.divide(k_f, k_r)
    return eqconstants
