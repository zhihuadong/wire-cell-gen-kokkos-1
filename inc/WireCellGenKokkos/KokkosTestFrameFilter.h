/**
 * To test if Kokkos works.
 */

#ifndef WIRECELL_GENKOKKOS_KOKKOSTESTFRAMEFILTER
#define WIRECELL_GENKOKKOS_KOKKOSTESTFRAMEFILTER

#include "WireCellIface/IConfigurable.h"
#include "WireCellIface/IFrameFilter.h"
#include "WireCellUtil/Logging.h"

namespace WireCell {
    namespace GenKokkos {

        class KokkosTestFrameFilter : public IFrameFilter, public IConfigurable {
           public:
            KokkosTestFrameFilter();
            virtual ~KokkosTestFrameFilter();

            /// working operation - interface from IFrameFilter
            /// executed when called by pgrapher
            virtual bool operator()(const IFrame::pointer &inframe, IFrame::pointer &outframe);

            /// interfaces from IConfigurable

            /// exeexecuted once at node creation
            virtual WireCell::Configuration default_configuration() const;

            /// executed once after node creation
            virtual void configure(const WireCell::Configuration &config);

           private:
            Configuration m_cfg;  /// copy of configuration

            /// SPD logger
            Log::logptr_t log;
        };
    }  // namespace GenKokkos
}  // namespace WireCell

#endif
