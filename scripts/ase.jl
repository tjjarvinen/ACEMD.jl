
using ACEmd
using ASEconvert
using LinearAlgebra: det
using PythonCall

calculator = pyimport("ase.calculators.calculator")
Calculator = calculator.Calculator

ACEcalculator = pytype("ACEcalculator", (Calculator,),[
    "implemented_properties" => ["energy", "forces"],

    pyfunc(
        name  = "__init__",
        function (self, potential, atoms=nothing)
            calculator.Calculator.__init__(self, atoms=atoms)

            self.potential = potential
            return
        end
    ),

    pyfunc(
        name = "calculate",
        function (self, atoms=nothing, properties=["energy"], system_changes=nothing)
            if "energy" in properties
                E = ace_energy(pyconvert(ACEpotential,self.potential), pyconvert(AbstractSystem, atoms))
                self.results["energy"] = ustrip(u"eV", E)
                self.results["free_energy"] = ustrip(u"eV", E)
            end

            if "forces" in properties
                F = ace_forces(pyconvert(ACEpotential,self.potential), pyconvert(AbstractSystem, atoms))
                conv = ustrip(u"eV/Å", F[1][1])
                self.results["forces"] = F * conv
            end

            if "stress" in properties
                virial = ace_virial(pyconvert(ACEpotential,self.potential), pyconvert(AbstractSystem, atoms))
                # stress = virial / volume
                V = (det ∘ hcat)( bounding_box(data)... )
                stress = virial/V
                ustress = ustrip.(u"eV/Å^2", stress)
                self.results["stress"] = ustress
            end
        end

    )
])

