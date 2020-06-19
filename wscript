#!/usr/bin/env python


TOP = '.'
APPNAME = 'Junk'

from waflib.extras import wcb
wcb.package_descriptions.append(("WCT", dict(
    incs=["WireCellUtil/Units.h"],
    libs=["WireCellUtil"], mandatory=True)))

wcb.package_descriptions.append(("KOKKOS", dict(
    incs=["Kokkos_Core.hpp"],
    libs=["kokkoscore"], mandatory=True)))

def options(opt):
    opt.load("wcb")

def configure(cfg):

    cfg.load("wcb")

    # boost 1.59 uses auto_ptr and GCC 5 deprecates it vociferously.
    cfg.env.CXXFLAGS += ['-Wno-deprecated-declarations']
    cfg.env.CXXFLAGS += ['-Wall', '-Wno-unused-local-typedefs', '-Wno-unused-function']
    cfg.env.CXXFLAGS += ['-fopenmp']
    # cfg.env.CXXFLAGS += ['-Wpedantic', '-Werror']


def build(bld):
    bld.load('wcb')
    bld.smplpkg('WireCellGenKokkos', use='WCT WireCellUtil WireCellIface WireCellGen JSONCPP BOOST EIGEN FFTW KOKKOS')
