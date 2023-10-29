// Deal.II Libraries
#include <algorithm>
#include <boost/iostreams/categories.hpp>
#include <deal.II/base/quadrature.h>
#include <deal.II/base/types.h>
#include <deal.II/base/function.h>

#include <deal.II/fe/fe_values.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparsity_pattern.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/data_out.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>

#include <deal.II/fe/fe_update_flags.h>
#include <deal.II/fe/fe_q.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/affine_constraints.h>

#include <random>
#include <unordered_map>
#include <fstream>

namespace cahnHilliard {
    using namespace dealii;

    template<int dim> class CahnHilliardEquation
    {
    public:
        CahnHilliardEquation(
            const std::unordered_map<std::string, double>  params,
            const double totalSimTime);
        void run();

    private:
        void setupSystem(const std::unordered_map<std::string, double> params,
                          const double                                  totalSimTime);
        void initializeValues();
        void constructRightHandEta();
        void constructRightHandC();
        void solveC();
        void solveEta();
        void outputResults() const;

        Triangulation<dim>  triangulation;
        FE_Q<dim>           fe;
        DoFHandler<dim>     dofHandler;

        AffineConstraints<double>   constraints;

        SparsityPattern         sparsityPattern;
        SparseMatrix<double>    massMatrix;
        SparseMatrix<double>    laplaceMatrix;
        SparseMatrix<double>    matrixC;
        SparseMatrix<double>    matrixEta;

        Vector<double> solutionC;
        Vector<double> solutionEta;
        Vector<double> oldSolutionC;
        Vector<double> systemRightHandSideC;
        Vector<double> systemRightHandSideEta;
        
        Vector<double>  cellRightHandSideEta;
        Vector<double>  cellRightHandSideC;

        double          timeStep;
        double          time;
        unsigned int    timestepNumber;
        double          totalSimTime;

        double eps;
    };

    template<int dim> CahnHilliardEquation<dim> 
        :: CahnHilliardEquation(std::unordered_map<std::string, double> params,
                                double totalSimTime)
            : fe(1)
            , dofHandler(triangulation)
            , timeStep(1. / 256.)
            , time(timeStep)
            , timestepNumber(1)
    {
        this->setupSystem(params,
                          totalSimTime);
    }

    template<int dim> void CahnHilliardEquation<dim> :: setupSystem(
        const std::unordered_map<std::string, double> params,
        const double                                  totalSimTime
    )
    {

        std::cout << "Passing parameters:" << std::endl;
        for(auto it=params.begin(); it!=params.end(); it++){
            std::cout   << "    "   << it->first
                        << ": "     << it->second
                        << std::endl;
        }

        this->eps = params.at("eps");
        this->totalSimTime = totalSimTime;
        
        std::cout << "Building mesh" << std::endl;

        GridGenerator::hyper_cube(
            this->triangulation,
            -1, 1,
            true
        );

        std::cout   << "Connecting nodes to neighbours due to periodic boundary"
                    << " conditions."
                    << std::endl;

        std::vector<GridTools::PeriodicFacePair<
            typename Triangulation<dim>::cell_iterator>
        > matchedPairsX;

        std::vector<GridTools::PeriodicFacePair<
            typename Triangulation<dim>::cell_iterator>
        > matchedPairsY;

        GridTools::collect_periodic_faces(this->triangulation,
                                          0, 1, 0, matchedPairsX);
        
        GridTools::collect_periodic_faces(this->triangulation,
                                          2, 3, 1, matchedPairsY);

        triangulation.add_periodicity(matchedPairsX);
        triangulation.add_periodicity(matchedPairsY);

        std::cout << "Neighbours updated to reflect periodicity" << std::endl;
    
        std::cout << "Refining grid" << std::endl;
        triangulation.refine_global(8);

        std::cout   << "Mesh generated...\n"
                    << "Active cells: " << triangulation.n_active_cells()
                    << std::endl;

        std::cout   << "Indexing degrees of freedom..."
                    << std::endl;

        this->dofHandler.distribute_dofs(fe);

        std::cout   << "Number of degrees of freedom: "
                    << dofHandler.n_dofs()
                    << std::endl;

        std::cout   << "Building sparsity pattern..."
                    << std::endl;

        std::vector<GridTools::PeriodicFacePair<
            typename DoFHandler<dim>::cell_iterator>
        > periodicity_vectorX;

        std::vector<GridTools::PeriodicFacePair<
            typename DoFHandler<dim>::cell_iterator>
        > periodicity_vectorY;

        GridTools::collect_periodic_faces(this->dofHandler,
                                          0,1,0,periodicity_vectorX);
        GridTools::collect_periodic_faces(this->dofHandler,
                                          2,3,1,periodicity_vectorY);

        DoFTools::make_periodicity_constraints<dim,dim>(periodicity_vectorX,
                                                        this->constraints);
        DoFTools::make_periodicity_constraints<dim,dim>(periodicity_vectorY,
                                                        this->constraints);

        constraints.close();

        DynamicSparsityPattern dsp(
            dofHandler.n_dofs(),
            dofHandler.n_dofs()
        );
        DoFTools::make_sparsity_pattern(dofHandler,
                                        dsp,
                                        this->constraints);
        sparsityPattern.copy_from(dsp);
        sparsityPattern.compress();

        std::cout   << "Reinitializing matices based on new pattern..."
                    << std::endl;

        massMatrix.reinit(sparsityPattern);
        laplaceMatrix.reinit(sparsityPattern);
        matrixC.reinit(sparsityPattern);
        matrixEta.reinit(sparsityPattern);

        std::cout   << "Filling entries for mass and laplace matrix..."
                    << std::endl;

        MatrixCreator::create_mass_matrix(
            dofHandler,
            QGauss<dim>(fe.degree+1),
            massMatrix
        );

        MatrixCreator::create_laplace_matrix(
            dofHandler,
            QGauss<dim>(fe.degree+1),
            massMatrix
        );

        std::cout   << "Initializing vectors..."
                    << std::endl;

        this->solutionC.reinit(dofHandler.n_dofs());
        this->solutionEta.reinit(dofHandler.n_dofs());
        this->oldSolutionC.reinit(dofHandler.n_dofs());
        this->systemRightHandSideC.reinit(dofHandler.n_dofs());
        this->systemRightHandSideEta.reinit(dofHandler.n_dofs());

        std::cout   << "\nCompleted construction\n"
                    << std::endl;

        std::cout   << "Building constraints from periodicity..."
                    << std::endl;


        std::cout   << "Periodicity constraints added"
                    << std::endl;
        
        // Initialize memory for cell values and RHS
        this->cellRightHandSideEta.reinit(fe.n_dofs_per_cell());
        this->cellRightHandSideC.reinit(fe.n_dofs_per_cell());

    }

    template<int dim> class InitialValuesC : public Function<dim>
    {
        public:
            InitialValuesC(double eps);
            virtual double value(
                const Point<dim> &p,
                const unsigned int component = 0
            ) const override;

        private:

            mutable std::default_random_engine  generator;
            mutable std::normal_distribution<double>    distribution;

            double eps;
    };

    template<int dim> InitialValuesC<dim>::InitialValuesC(double eps)
        : generator()
        , distribution(0,0.25)
        , eps(eps)
    {}
    
    template<int dim> double InitialValuesC<dim> :: value(
        const Point<dim> &p,
        const unsigned int i /*component*/
    ) const
    {
        return std::tanh(
            (p.norm()-0.25) / (std::sqrt(2)*this->eps)
        ); 
    }

    template<int dim> void CahnHilliardEquation<dim> :: initializeValues()
    {   
       
        std::cout   << "Initializing values for C" << std::endl;

        VectorTools::project(this->dofHandler,
                             this->constraints,
                             QGauss<dim>(fe.degree + 1),
                             InitialValuesC<dim>(this->eps),
                             this->oldSolutionC);
        this->constraints.distribute(this->oldSolutionC);

        double* maxC = std::max_element(oldSolutionC.begin(),
                                        oldSolutionC.end());
        double* minC = std::min_element(oldSolutionC.begin(),
                                        oldSolutionC.end());

        std::cout   << "Initial values propagated:\n"
                    << "    Range: (" 
                        << *minC << ", " 
                        << *maxC
                    << ")" 
                    << std::endl;

    }

    template<int dim> void CahnHilliardEquation<dim>
        :: solveEta()
    {   
        SolverControl               solverControl(
                                        1000,
                                        1e-8 * systemRightHandSideEta.l2_norm()
                                    );
        SolverCG<Vector<double>>    cg(solverControl);

        this->constraints.condense(this->massMatrix,
                                   this->systemRightHandSideEta);
        cg.solve(
            this->massMatrix,
            this->solutionEta,
            this->systemRightHandSideEta,
            PreconditionIdentity()
        );
        this->constraints.distribute(this->solutionEta);

        std::cout   << "    Eta solved: "
                    << solverControl.last_step()
                    << " CG iterations."
                    << std::endl;
    }

    template<int dim> void CahnHilliardEquation<dim>
        :: solveC()
    {   
        SolverControl               solverControl(
                                        1000,
                                        1e-8 * systemRightHandSideC.l2_norm()
                                    );
        SolverCG<Vector<double>>    cg(solverControl);

        this->constraints.condense(this->massMatrix,
                                   this->systemRightHandSideC);
        cg.solve(
            this->massMatrix,
            this->solutionC,
            this->systemRightHandSideC,
            PreconditionIdentity()
        );
        this->constraints.distribute(this->solutionC);

        std::cout   << "    C solved: "
                    << solverControl.last_step()
                    << " CG iterations."
                    << std::endl;
    }
    
    template<int dim> void CahnHilliardEquation<dim> 
        :: outputResults() const
    {

        DataOut<dim> dataOut;

        dataOut.attach_dof_handler(this->dofHandler);
        dataOut.add_data_vector(this->solutionC, "C");
        dataOut.add_data_vector(this->solutionEta, "Eta");
        dataOut.build_patches();

        const std::string filename = ("data/solution-" 
                                     + std::to_string(this->timestepNumber) 
                                     + ".vtu");

        DataOutBase::VtkFlags vtk_flags;
        vtk_flags.compression_level = DataOutBase::VtkFlags::ZlibCompressionLevel::best_speed;
        dataOut.set_flags(vtk_flags);

        std::ofstream output(filename);
        dataOut.write_vtu(output);

    };

    template<int dim> void CahnHilliardEquation<dim>
        :: run()
    {
        this->initializeValues();

        QGauss<dim>     quadFormula(fe.degree + 1);
        FEValues<dim>   feValues(fe,
                                 quadFormula,
                                 update_values | 
                                 update_JxW_values |
                                 update_gradients);

        std::vector<types::global_dof_index> 
            local_dof_indices(fe.n_dofs_per_cell());

        std::vector<double> cellValuesC(quadFormula.size());
        std::vector<Tensor<1,dim>> cellGradEta(quadFormula.size());

        Vector<double> temp(dofHandler.n_dofs());

        for(; this->time < this->totalSimTime; ++this->timestepNumber)
        {
            this->time += this->timeStep;

            std::cout   << "\rCurrent time: " << this->time << std::flush; 
            std::cout   << "    Constructing RHS for eta^n" << std::endl;

            // systemRightHandSideEta = AC^n
            laplaceMatrix.vmult(systemRightHandSideEta, this->oldSolutionC);

            this->systemRightHandSideEta *= pow(this->eps,2);

            for(const auto &cell : this->dofHandler.active_cell_iterators())
            {
                feValues.reinit(cell);
                feValues.get_function_values(oldSolutionC,
                                             cellValuesC);

                cell->get_dof_indices(local_dof_indices);
                for(const unsigned int qIndex : feValues.quadrature_point_indices())
                {
                    double Cx = cellValuesC[qIndex];
                    
                    if (std::abs(Cx) > 2){
                        std::cout << std::endl;
                        std::cerr << "Value has fallen outside of viable range";
                        std::cout << std::endl;
                        assert(std::abs(Cx) <= 2);
                    }
                    for(const unsigned int i : feValues.dof_indices())
                    {
                        // systemRightHandSide -= (phi_i, F(C^n))
                        // where F(C^n) = ( C^n (1 - (C^n)^2) )
                        this->systemRightHandSideEta(local_dof_indices[i])
                            += feValues.shape_value(i, qIndex) 
                                * (Cx * (pow(Cx,2) - 1))
                                * feValues.JxW(qIndex);
                    }
                }
            }
            
            std::cout << "    Solving for eta^n" << std::endl;
            solveEta();
            
            std::cout << "    Constructing RHS for c^{n+1}" << std::endl;

            massMatrix.vmult(systemRightHandSideC, this->oldSolutionC);

            for(const auto &cell : this->dofHandler.active_cell_iterators())
            {
                feValues.reinit(cell);

                feValues.get_function_values(oldSolutionC,
                                             cellValuesC);
                feValues.get_function_gradients(solutionEta,
                                                cellGradEta);

                cell->get_dof_indices(local_dof_indices);
                for(const unsigned int qIndex : feValues.quadrature_point_indices())
                {
                    double Cx       = cellValuesC[qIndex];
                    auto GradEtaX   = cellGradEta[qIndex];

                    for(const unsigned int i : feValues.dof_indices())
                    {
                        this->systemRightHandSideC(local_dof_indices[i])
                            -= (feValues.shape_grad(i, qIndex)
                                * (1-pow(Cx,2))
                                * GradEtaX) * feValues.JxW(qIndex) * this->timeStep;
                                
                    }
                }
            }

            std::cout << "    Solving for c^{n+1}" << std::endl;
            solveC();
 
 

            this->oldSolutionC = this->solutionC;

            if (this->timestepNumber % 5 == 0){

                double maxC = *std::max_element(this->solutionC.begin(),
                                                this->solutionC.end());
                double minC = *std::min_element(this->solutionC.begin(),
                                                this->solutionC.end());
                double maxEta = *std::max_element(this->solutionEta.begin(),
                                                  this->solutionEta.end());
                double minEta = *std::min_element(this->solutionEta.begin(),
                                                  this->solutionEta.end());

                std::cout   << "    C Range: (" 
                                << minC  << ", " 
                                << maxC 
                            << ")" << std::endl;
                std::cout   << "    Eta Range: (" 
                                << minEta  << ", " 
                                << maxEta 
                            << ")" << std::endl;
                outputResults();
            }

        }

    }

}


int main(){ 
    std::cout   << "Running" << std::endl << std::endl;

    std::unordered_map<std::string, double> params;

    params["eps"] = 1e-2;

    double totalSimTime = 10;

    cahnHilliard::CahnHilliardEquation<2> cahnHilliard(params, totalSimTime);
    cahnHilliard.run();

    std::cout << "Completed" << std::endl;

    return 0;
}
