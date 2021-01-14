#ifndef WIRECELL_GENKOKKOS_KOKKOSENV
#define WIRECELL_GENKOKKOS_KOKKOSENV

#include "WireCellIface/ITerminal.h"

namespace WireCell {
    namespace GenKokkos {
        class KokkosEnv : public WireCell::ITerminal {
           private:
            static bool kokkos_initialized;

           public:
            KokkosEnv();
            virtual ~KokkosEnv();
            virtual void finalize();
        };

    }  // namespace GenKokkos
}  // namespace WireCell

#endif