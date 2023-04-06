
function neigsz!(tmp, nlist::PairList, at::Atoms, i::Integer)
    # from JuLIP
    j, R = neigs!(tmp.R, nlist, i)
    Z = tmp.Z
    for n in eachindex(j)
       Z[n] = at.Z[j[n]]
    end
    return j, R, (@view Z[1:length(j)])
 end
 
 function neigsz(nlist::PairList, at::Atoms, i::Integer)
    # from JuLIP
    j, R = NeighbourLists.neigs(nlist, i)
    return j, R, at.Z[j]
 end


 function load_ace_model(fname; old_format=false)
    pot_tmp = load_dict(fname)["IP"]
    pot = read_dict(pot_tmp)
    if old_format
        return pot
    else
        return pot.components
    end
 end