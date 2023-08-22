module ACE_Molly_ext

import ACE1: AtomicNumber
using ACEmd
using Molly


function Molly.forces(
        acp::ACEpotential,
        sys,
        neighbors=nothing;
        n_threads=nothing,
        executor=ThreadedEx()
    )
    return ace_forces(acp, sys; executor=executor) 
end


function Molly.potential_energy(
        acp::ACEpotential,
        sys,
        neighbors=nothing;
        n_threads=nothing,
        executor=ThreadedEx()
    )
    return ace_energy(acp, sys; executor=executor) 
end
 

function ACEmd._atomic_number(sys::Molly.System, i) 
    return AtomicNumber( sys.atoms_data[i].Z )
end

end # Module
