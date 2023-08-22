
function neigsz!(tmp, nlist::PairList, at::ACE1.Atoms, i::Integer)
    # from JuLIP
    j, R = neigs!(tmp.R, nlist, i)
    Z = tmp.Z
    for n in eachindex(j)
       Z[n] = at.Z[j[n]]
    end
    return j, R, (@view Z[1:length(j)])
end
 
function neigsz(nlist::PairList, at, i::Integer)
    j, R = NeighbourLists.neigs(nlist, i)
    Z = map(j) do k
        _atomic_number(at, k)
    end
    return j, R, Z 
end

"""
    load_ace_model(fname::AbstractString; Kwargs)
    load_ace_model(potential: Kwargs)

Used to load potential from `json` or `yml` files.
By default this adds units for the potential. You can use
Kwargs to control what units are used. The default units
are defined in `ACEmd.default_energy` and `ACEmd.default_length`.

You can also use this to convert non unit holding potentials to
potentials that have units. Like e.g. when you have fitted a new
potential it does not have unit. You can then use this function to
wrap units for it.

# Kwargs
- `energy_unit=default_energy`  :  energy unit for the potential
- `length_unit=default_length`  :  lenght unit used for the force and atomic system
- `cutoff_unit=default_length`  :  unit in which cuttoff radius has been defined
"""
function load_ace_model(fname::AbstractString;
        old_format=false,
        energy_unit=default_energy,
        length_unit=default_length,
        cutoff_unit=default_length
    )
    pot_tmp = load_dict(fname)
    if haskey(pot_tmp, "IP")
        pot = read_dict(pot_tmp["IP"])
    elseif haskey(pot_tmp, "potential")
        pot = read_dict(pot_tmp["potential"])
    else
        error("Potential format not recognised")
    end
    if old_format
        return pot
    else
        return ACEpotential(pot.components; energy_unit=energy_unit, length_unit=length_unit, cutoff_unit=cutoff_unit)
    end
end


function load_ace_model(pot;
        energy_unit=default_energy,
        length_unit=default_length,
        cutoff_unit=default_length
    )
    return ACEpotential(pot.components; energy_unit=energy_unit, length_unit=length_unit, cutoff_unit=cutoff_unit)
end


function neighborlist(ab, cutoff; length_unit=default_length, kwargs...)
    cell = ustrip.(length_unit, hcat( bounding_box(ab)... ) )
    pbc = map( boundary_conditions(ab) ) do x
        x == Periodic()
    end
    r = map( 1:length(ab)) do i
        # Need to have SVector here for PairList to work
        # if position does not give SVector
        SVector( ustrip.(length_unit, position(ab,i))...)
    end
    nlist = PairList(r, ustrip(length_unit, cutoff), cell, pbc; int_type=Int)
    return nlist
end


function get_cutoff(V; cutoff_unit=default_length)
    return ACE1.cutoff(V) * cutoff_unit
end

function get_cutoff(V::ACEpotential; kwargs...)
    c = map( V ) do v
        if typeof(v) <: ACE1.OneBody
            return 0
        else
            return ACE1.cutoff(v)
        end
    end
    return maximum( c )  * V.cutoff_unit
end



 ## CellListMap

function neighborlist(at::ACE1.Atoms, cutoff; kwargs...)
    return ACE1.neighbourlist(at, ustrip(u"Å", cutoff); kwargs...)
end


function neighborlist_clm(ab::AbstractSystem, cutoff; kwargs...)
    function push_pair!(i, j, x, y, d2, pairs, cutoff) 
        d = sqrt(d2)
        if d < cutoff
            push!(pairs, (i, j, ustrip.(y-x)))
        end
        return pairs
    end
    function reduce_pairs(pairs, pairs_threaded)
        for i in eachindex(pairs_threaded)
            append!(pairs, pairs_threaded[i])
        end
        return pairs
    end
    ucutoff = cutoff*u"Å"  # ACE cutoff is in Å
    tmp = bounding_box(ab)
    cell = [ tmp[i][i] for i in eachindex(tmp) ]

    box = Box(cell, ucutoff)
    
    #TODO allow dynamic types here
    pairs = Tuple{Int,Int, SVector{3, Float64}}[]

    cl = CellList(position(ab), box, parallel=true)

    map_pairwise!(
        (x, y, i, j, d2, pairs) -> push_pair!(i, j, x, y, d2, pairs, ucutoff),
        pairs,box,cl,
        reduce=reduce_pairs,
        parallel=true
    )
    return pairs
end


function neigsz(list, ab, i)
    T = typeof( list[begin][3] )
    R = T[]
    j = Int[]
    Z = AtomicNumber[] # JuLIP from ACE1

    for x in list
        if x[1] == i
            push!(R, x[3])
            push!(j, x[2])
            push!(Z, _atomic_number(ab,x[2]))
        elseif x[2] == i
            push!(R, -x[3])
            push!(j, x[1])
            push!(Z, _atomic_number(ab,x[1]))
        end
    end
    return j, R, Z
end



_atomic_number(at::ACE1.Atoms, i)          = at.Z[i]
_atomic_number(ab::AbstractSystem, i) = ACE1.AtomicNumber(AtomsBase.atomic_number(ab,i))