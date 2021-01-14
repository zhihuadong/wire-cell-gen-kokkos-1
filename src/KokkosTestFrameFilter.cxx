#include "WireCellGenKokkos/KokkosTestFrameFilter.h"

#include "WireCellUtil/NamedFactory.h"

WIRECELL_FACTORY(KokkosTestFrameFilter, WireCell::GenKokkos::KokkosTestFrameFilter, WireCell::IFrameFilter,
                 WireCell::IConfigurable)