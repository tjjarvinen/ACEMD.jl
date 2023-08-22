
const default_length = u"Å"
const default_energy = u"eV"

struct ACEpotential{TE,TL,TC}
    potentials::Vector{ACE1.AbstractCalculator}
    energy_unit::TE
    length_unit::TL
    cutoff_unit::TC
    function ACEpotential(potentials; energy_unit=default_energy, length_unit=default_length, cutoff_unit=length_unit)
        @assert dimension(energy_unit) == dimension(u"J")
        @assert dimension(length_unit) == dimension(u"m")
        @assert dimension(cutoff_unit) == dimension(u"m")
        new{typeof(energy_unit), typeof(length_unit), typeof(cutoff_unit)}(potentials, energy_unit, length_unit, cutoff_unit)
    end
end


function Base.iterate(acep::ACEpotential, state::Int=1)
    if 0 < state <= length( acep )
        return acep.potentials[state], state + 1
    else
        return nothing
    end
end

Base.length(acep::ACEpotential) = length(acep.potentials)

Base.getindex(acep::ACEpotential, i) = acep.potentials[i]

function Base.show(io::IO, ::MIME"text/plain", acep::ACEpotential)
    print(io, "ACE potential consisting of ", length(acep), " subpotentials")
end