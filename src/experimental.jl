import Base.Threads.@spawn


function energy_floops(calc, at::Atoms; domain=1:length(at), executor=ThreadedEx())
    nlist = neighbourlist(at, cutoff(calc))
    @floop executor for i in domain
        _, R, Z = neigsz(nlist, at, i)
        @reduce (Etot += ace_evaluate(calc, R, Z, at.Z[i]) )
    end
    return Etot
end


function energy_tasks(calc, at::Atoms; ntasks=1)
    nlist = neighbourlist(at, cutoff(calc))
    Δ = (Int ∘ floor)( length(at) / ntasks )
    tasks = map( 1:ntasks ) do i
        s = 1+(i-1)*Δ : i*Δ
        @spawn energy_nonthreaded(calc, at::Atoms, nlist; domain=s)
    end
    Etot = sum(tasks) do t
        fetch(t)
    end
    return Etot
end

function energy_nonthreaded!(tmp, calc, at::Atoms; domain=1:length(at))
    # tmp = ACE1.alloc_temp(calc, at)
    nlist = neighbourlist(at, cutoff(calc))
    Etot = sum( domain ) do i
        _, R, Z = neigsz!(tmp, nlist, at, i)
        ace_evaluate!(tmp, calc, R, Z, at.Z[i]) 
    end
    return Etot
end 


function energy_nonthreaded(calc, at::Atoms; domain=1:length(at))
    nlist = neighbourlist(at, cutoff(calc))
    Etot = sum( domain ) do i
        _, R, Z = neigsz(nlist, at, i)
        ace_evaluate(calc, R, Z, at.Z[i])
    end
    return Etot
end

function energy_nonthreaded(calc, at::Atoms, nlist; domain=1:length(at))
    #nlist = neighbourlist(at, cutoff(calc))
    Etot = sum( domain ) do i
        _, R, Z = neigsz(nlist, at, i)
        ace_evaluate(calc, R, Z, at.Z[i])
    end
    return Etot
end

##

function ace_forces_old(V, at; domain=1:length(at), executor=ThreadedEx())
    nlist = neighborlist(at, cutoff(V))
    F = Folds.sum( domain, executor ) do i
        j, R, Z = neigsz(nlist, at, i)
        _, tmp = ace_evaluate_d(V, R, Z, _atomic_number(at,i))

        #TODO make this faster
        f = zeros(eltype(tmp.dV), length(at))
        for k in eachindex(j)
            f[j[k]] -= tmp.dV[k]
            f[i]    += tmp.dV[k]
        end
        f
    end
    return F
end