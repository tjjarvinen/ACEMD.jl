
function ace_energy(calc, at; domain=1:length(at), executor=ThreadedEx())
    nlist = neighborlist(at, cutoff(calc); storelist=false)
    Etot = Folds.sum( domain, executor ) do i
        _, R, Z = neigsz(nlist, at, i)
        ace_evaluate(calc, R, Z, _atomic_number(at,i))
    end
    return Etot
end

function ace_energy(V::OneBody, at::Atoms; domain=1:length(at), executor=nothing)
    E = sum( domain ) do i
        ACE1.evaluate(V, chemical_symbol(at.Z[i]) )
    end
    return E
end

function ace_energy(V::OneBody, as::AbstractSystem; domain=1:length(as), executor=nothing)
    E = sum( domain ) do i
        ACE1.evaluate(V, atomic_symbol(as, i) )
    end
    return E
end

# Generate interface for multiple potentials
for ace_method in [ :ace_energy, :ace_forces, :ace_virial ]
    @eval begin
        function $ace_method(calc::Union{AbstractArray, ACEpotential}, at; domain=1:length(at), executor=ThreadedEx())
            tmp = asyncmap( calc ) do V
                $ace_method(V, at; domain=domain, executor=executor)
            end
            return sum(tmp)
        end
    end
end




## forces

function ace_forces(V, at; domain=1:length(at), executor=ThreadedEx())
    # functions to reduce allocations during reduction
    function _reduce(s::AbstractVector, a)
        for i in eachindex(a[2])
            s[a[2][i]] -= a[3][i]
        end
        s[a[1]] += sum(a[3])
        return s
    end
    function _reduce(s::AbstractVector, a::AbstractVector)
        return s .+ a
    end
    nlist = neighborlist(at, cutoff(V))
    F = Folds.mapreduce( _reduce,  domain, executor; init=zeros(SVector{3, Float64}, length(at)) ) do i
        j, R, Z = neigsz(nlist, at, i)
        _, tmp = ace_evaluate_d(V, R, Z, _atomic_number(at,i))

        i, j, tmp.dV
    end
    return F
end


function ace_forces(::OneBody, at::Atoms; kwargs...)
    T = (eltype ∘ eltype)(at.X)
    F = zeros(SVector{3,T}, length(at)  )
    return F
end

function ace_forces(::OneBody, as::AbstractSystem; kwargs...)
    T = eltype( ustrip.( position(as, 1) )  )
    #F = [ SVector{3}( zeros(T, 3) ) for _ in 1:length(as) ]
    F = zeros(SVector{3,T}, length(as)  )
    return F
end


## virial

function ace_virial(V, at; domain=1:length(at), executor=ThreadedEx())
    nlist = neighborlist(at, cutoff(V))
    vir = Folds.sum( domain, executor ) do i
        j, R, Z = neigsz(nlist, at, i)
        _, tmp = ace_evaluate_d(V, R, Z, _atomic_number(at,i))
        site_virial = -sum( zip(R, tmp.dV) ) do (Rⱼ, dVⱼ)
            dVⱼ * Rⱼ'
        end
        site_virial
    end
    return vir
end

function ace_virial(::OneBody, at::Atoms; kwargs...)
    T = (eltype ∘ eltype)(at.X)
    return SMatrix{3,3}(zeros(T, 3,3))
end

function ace_virial(::OneBody, as::AbstractSystem; kwargs...)
    T = eltype( ustrip.( position( as[begin] ) )  )
    return SMatrix{3,3}(zeros(T, 3,3))
end


## Combinations
# these will be optimized later

function ace_energy_forces(pot, data; domain=1:length(data), executor=ThreadedEx())
    E = ace_energy(pot, data; domain=domain, executor=executor)
    F = ace_forces(pot, data; domain=domain, executor=executor)
    return Dict("energy"=>E, "forces"=>F)
end


function ace_energy_forces_virial(pot, data; domain=1:length(data), executor=ThreadedEx())
    E = ace_energy(pot, data; domain=domain, executor=executor)
    F = ace_forces(pot, data; domain=domain, executor=executor)
    V = ace_virial(pot, data; domain=domain, executor=executor)
    return Dict("energy"=>E, "forces"=>F, "virial"=>V)
end

function ace_forces_virial(pot, data; domain=1:length(data), executor=ThreadedEx())
    F = ace_forces(pot, data; domain=domain, executor=executor)
    V = ace_virial(pot, data; domain=domain, executor=executor)
    return Dict("forces"=>F, "virial"=>V)
end