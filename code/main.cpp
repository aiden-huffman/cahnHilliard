// Deal.II Libraries
#include <algorithm>
#include <boost/iostreams/categories.hpp>
#include <deal.II/base/quadrature.h>
#include <deal.II/base/types.h>
#include <deal.II/base/function.h>

#include <deal.II/fe/fe_values.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/linear_operator.h>
#include <deal.II/lac/schur_complement.h>
#include <deal.II/lac/sparsity_pattern.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/data_out.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/fe/fe_update_flags.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/precondition_selector.h>
#include <deal.II/lac/sparse_ilu.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/linear_operator_tools.h>
#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/block_sparse_matrix.h>

#include <deal.II/dofs/dof_renumbering.h>

#include <ostream>
#include <random>
#include <unordered_map>
#include <fstream>

namespace cahnHilliard {
    using namespace dealii;

template<int dim>
class InitialValuesC : public Function<dim>
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

template<int dim> 
InitialValuesC<dim>::InitialValuesC(double eps)
    : eps(eps)
{}
    
template<int dim>
double InitialValuesC<dim> :: value(
    const Point<dim> &p,
    const unsigned int /*component*/
) const
{
    return std::tanh(
        (p.norm()-0.25) / (std::sqrt(2)*this->eps)
    ); 
}

template<int dim>
class CahnHilliardEquation
{
public:
    CahnHilliardEquation();
    void run(
        const std::unordered_map<std::string, double> params,
        const double                                  totalSimTime
    );

private:
    void setupSystem(
            const std::unordered_map<std::string, double> params,
            const double                                  totalSimTime
    );
    void setupTriang();
    void setupDoFs();
    void reinitLinearSystem();
    void initializeValues();

    void assembleSystem();
    void updateRHS();
    void solveSystem();

    void outputResults() const;
    
    uint                degree;
    Triangulation<dim>  triangulation;
    FE_Q<dim>           fe;
    QGauss<dim>         quad_formula;
    FEValues<dim>       fe_values;
    DoFHandler<dim>     dof_handler;

    AffineConstraints<double>   constraints;

    SparsityPattern         sparsity_pattern;
    SparseMatrix<double>    mass_matrix;
    SparseMatrix<double>    laplace_matrix;

    Vector<double>          phi_solution;
    Vector<double>          phi_old_solution;
    Vector<double>          phi_rhs;

    Vector<double>          eta_solution;
    Vector<double>          eta_rhs;

    Vector<double>      local_phi_rhs;
    Vector<double>      local_eta_rhs;

    double          timestep;
    double          time;
    unsigned int    timestep_number;
    double          totalSimTime;

    double eps;
};

template<int dim> 
CahnHilliardEquation<dim> :: CahnHilliardEquation()
    : degree(1)
    , fe(FE_Q<dim>(degree))
    , quad_formula(degree+1)
    , fe_values(this->fe,
                this->quad_formula,
                update_values |
                update_gradients |
                update_JxW_values)
    , dof_handler(triangulation)
    , timestep(1e-7)
    , time(timestep)
    , timestep_number(1)
{}

template<int dim>
void CahnHilliardEquation<dim> :: setupTriang(){

    std::cout << "Building mesh" << std::endl;

    GridGenerator::hyper_cube(
        this->triangulation,
        -1, 1,
        true
    );

    std::cout   << "Connecting nodes to neighbours due to periodic boundary"
                << " conditions."
                << std::endl;
    
    if(dim == 2){

        std::vector<GridTools::PeriodicFacePair<
            typename Triangulation<dim>::cell_iterator>
        > matched_pairs_X;

        std::vector<GridTools::PeriodicFacePair<
            typename Triangulation<dim>::cell_iterator>
        > matched_pairs_Y;

        GridTools::collect_periodic_faces(this->triangulation,
                                          0, 1, 0, matched_pairs_X);
        
        GridTools::collect_periodic_faces(this->triangulation,
                                          2, 3, 1, matched_pairs_Y);

        triangulation.add_periodicity(matched_pairs_X);
        triangulation.add_periodicity(matched_pairs_Y);

    } else if (dim == 3) {

        std::vector<GridTools::PeriodicFacePair<
            typename Triangulation<dim>::cell_iterator>
        > matched_pairs_X;

        std::vector<GridTools::PeriodicFacePair<
            typename Triangulation<dim>::cell_iterator>
        > matched_pairs_Y;
        
        std::vector<GridTools::PeriodicFacePair<
            typename Triangulation<dim>::cell_iterator>
        > matched_pairs_Z;

        GridTools::collect_periodic_faces(this->triangulation,
                                          0, 1, 0, matched_pairs_X);
        
        GridTools::collect_periodic_faces(this->triangulation,
                                          2, 3, 1, matched_pairs_Y);

        GridTools::collect_periodic_faces(this->triangulation,
                                          4, 5, 2, matched_pairs_Z);

        triangulation.add_periodicity(matched_pairs_X);
        triangulation.add_periodicity(matched_pairs_Y);
        triangulation.add_periodicity(matched_pairs_Z);
    }

    std::cout << "Neighbours updated to reflect periodicity" << std::endl;

    std::cout << "Refining grid" << std::endl;
    triangulation.refine_global(8);

    std::cout   << "Mesh generated...\n"
                << "Active cells: " << triangulation.n_active_cells()
                << std::endl;

}

template<int dim>
void CahnHilliardEquation<dim> :: setupDoFs()
{

    std::cout   << "Indexing degrees of freedom..."
                << std::endl;

    this->dof_handler.distribute_dofs(fe);
    DoFRenumbering::Cuthill_McKee(this->dof_handler);
     
    std::cout   << "Number of degrees of freedom: "
                << dof_handler.n_dofs()
                << std::endl;

    std::cout   << "Adding periodicity considerations to degrees of freedom"
                << " and constraints"
                << std::endl;

    std::vector<GridTools::PeriodicFacePair<
        typename DoFHandler<dim>::cell_iterator>
    > periodicity_vectorX;

    std::vector<GridTools::PeriodicFacePair<
        typename DoFHandler<dim>::cell_iterator>
    > periodicity_vectorY;

    GridTools::collect_periodic_faces(this->dof_handler,
                                      0,1,0,periodicity_vectorX);
    GridTools::collect_periodic_faces(this->dof_handler,
                                      2,3,1,periodicity_vectorY);

    DoFTools::make_periodicity_constraints<dim,dim>(periodicity_vectorX,
                                                    this->constraints);
    DoFTools::make_periodicity_constraints<dim,dim>(periodicity_vectorY,
                                                    this->constraints);

    std::cout   << "Closing constraints" << std::endl;
    this->constraints.close();

    std::cout   << "Building sparsity pattern..."
                << std::endl;

    DynamicSparsityPattern dsp(this->dof_handler.n_dofs());
    DoFTools::make_sparsity_pattern(
        this->dof_handler,
        dsp,
        this->constraints);

    sparsity_pattern.copy_from(dsp);
    sparsity_pattern.compress();
    
}

template<int dim>
void CahnHilliardEquation<dim> :: reinitLinearSystem()
{
    std::cout   << "Reinitializing the objects for the linear system"
                << std::endl;

    const std::vector<types::global_dof_index> dofs_per_component =
        DoFTools::count_dofs_per_fe_component(this->dof_handler);
    
    this->mass_matrix.reinit(sparsity_pattern);
    this->laplace_matrix.reinit(sparsity_pattern);

    MatrixTools::create_mass_matrix(this->dof_handler,
                                    this->quad_formula,
                                    this->mass_matrix);
    MatrixTools::create_laplace_matrix(this->dof_handler,
                                       this->quad_formula,
                                       this->laplace_matrix);

    this->phi_solution.reinit(this->dof_handler.n_dofs());
    this->phi_old_solution.reinit(this->dof_handler.n_dofs());
    this->phi_rhs.reinit(this->dof_handler.n_dofs());

    this->eta_solution.reinit(this->dof_handler.n_dofs());
    this->eta_rhs.reinit(this->dof_handler.n_dofs());

    this->local_phi_rhs.reinit(this->fe.dofs_per_cell);
    this->local_eta_rhs.reinit(this->fe.dofs_per_cell);
    
    this->constraints.condense(this->mass_matrix);
    this->constraints.condense(this->laplace_matrix);

}

template<int dim>
void CahnHilliardEquation<dim> :: setupSystem(
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
    
    this->setupTriang();
    this->setupDoFs();
    this->reinitLinearSystem();
}

template<int dim>
void CahnHilliardEquation<dim> :: initializeValues()
{   
   
    std::cout   << "Initializing values for phi" << std::endl;

    VectorTools::project(this->dof_handler,
                         this->constraints,
                         QGauss<dim>(fe.degree + 1),
                         InitialValuesC<dim>(this->eps),
                         this->phi_old_solution);

    this->constraints.distribute(this->phi_old_solution);
    
    auto phi_range = std::minmax_element(this->phi_old_solution.begin(),
                                         this->phi_old_solution.end());

    std::cout   << "Initial values propagated:\n"
                << "    Phi Range: (" 
                    << *phi_range.first << ", " 
                    << *phi_range.second
                << ")" 
                << std::endl;

}

template<int dim>
void CahnHilliardEquation<dim> :: assembleSystem()
{

    std::cout << "Assembling system" << std::endl;

    this->phi_rhs = 0;
    this->eta_rhs = 0;

    this->mass_matrix.vmult(this->phi_rhs, this->phi_old_solution);

    const unsigned int dofs_per_cell = this->fe.n_dofs_per_cell();


    std::vector<double>          cell_old_phi_values(this->quad_formula.size());
    std::vector<Tensor<1,dim>>   cell_old_phi_grad(this->quad_formula.size());
    std::vector<Tensor<1,dim>>   cell_old_eta_grad(this->quad_formula.size());

    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    const FEValuesExtractors::Scalar    phi(0);
    const FEValuesExtractors::Scalar    eta(1);

    for(const auto &cell : dof_handler.active_cell_iterators())
    {

        this->fe_values.reinit(cell);
        this->local_eta_rhs = 0;

        cell->get_dof_indices(local_dof_indices);

       this->fe_values.get_function_values(
            this->phi_old_solution,
            cell_old_phi_values
        ); 

        for(uint q_index = 0 ;  q_index < this->quad_formula.size(); q_index++)
        {   

            double          phi_old_x       = cell_old_phi_values[q_index];

            for(uint i = 0; i < dofs_per_cell; i++)
            {
                 
                local_eta_rhs(i)    +=  this->fe_values.shape_value(i,q_index)
                                    *   pow(phi_old_x,3)
                                    *   this->fe_values.JxW(q_index);

                this->constraints.distribute_local_to_global(
                    local_eta_rhs,
                    local_dof_indices,
                    this->eta_rhs
                );
            }
        }
    }

    std::cout << "Assembly completed" << std::endl;
}

template<int dim>
void CahnHilliardEquation<dim> :: solveSystem()
{

    std::cout << "Solving system" << std::endl;

    SolverControl               solverControlInner(
                                    5000,
                                    1e-10 * this->eta_rhs.l2_norm()
                                );
    SolverCG<Vector<double>>    solverInner(solverControlInner);

    SolverControl               solverControlOuter(
                                    10000,
                                    1e-8 
                                );
    SolverGMRES<Vector<double>>    solverOuter(solverControlOuter);
   
    
    const auto M = linear_operator(this->mass_matrix);
    const auto L = linear_operator(this->laplace_matrix);

    auto A = M;
    auto B = this->timestep * L;
    auto C = M + pow(this->eps, 2) * L;
    auto D = M;

    SparseILU<double> precon_A;
    precon_A.initialize(this->mass_matrix);
    auto A_inv = inverse_operator(M, solverInner, precon_A);

    auto S = schur_complement(A_inv, B, C, D);

    SparseILU<double> precon_S;
    precon_S.initialize(this->laplace_matrix);
    auto S_inv = inverse_operator(S, solverOuter, this->mass_matrix);
    auto rhs = condense_schur_rhs(A_inv, C, this->phi_rhs, this->eta_rhs);

    this->eta_solution = S_inv * rhs;
    this->constraints.distribute(this->eta_solution);

    this->phi_solution = postprocess_schur_solution(
        A_inv, B, this->eta_solution, this->phi_rhs
    );
    this->constraints.distribute(this->phi_solution);

    auto phi_range  = std::minmax_element(this->phi_solution.begin(),
                                          this->phi_solution.end());
    auto eta_range  = std::minmax_element(this->eta_solution.begin(),
                                          this->eta_solution.end());

    std::cout   << "Result for phi:\n"
                << "    Phi Range: (" 
                    << *phi_range.first << ", " 
                    << *phi_range.second
                << ")" 
                << std::endl;

    std::cout   << "Results for eta:\n"
                << "    Eta Range: (" 
                    << *eta_range.first << ", " 
                    << *eta_range.second
                << ")" 
                << std::endl;

    this->phi_old_solution = this->phi_solution;
}

template<int dim>
void CahnHilliardEquation<dim> :: outputResults() const
{

    DataOut<dim> dataOut;

    std::vector<DataComponentInterpretation::DataComponentInterpretation>
        interpretation(2,DataComponentInterpretation::component_is_scalar);
    std::vector<std::string> solution_names = {"phi", "eta"};
    std::vector<std::string> rhs_names = {"phi_rhs", "eta_rhs"};
    std::vector<std::string> old_names = {"phi_old", "eta_old"};

    dataOut.add_data_vector(this->dof_handler,
                            this->phi_solution,
                            "phi");
    dataOut.add_data_vector(this->dof_handler,
                            this->eta_solution,
                            "eta");
    dataOut.build_patches();

    const std::string filename = ("data/solution-" 
                                 + std::to_string(this->timestep_number) 
                                 + ".vtu");

    DataOutBase::VtkFlags vtk_flags;
    vtk_flags.compression_level = DataOutBase
        ::VtkFlags
        ::ZlibCompressionLevel
        ::best_speed;
    dataOut.set_flags(vtk_flags);

    std::ofstream output(filename);
    dataOut.write_vtu(output);
    
};

template<int dim> 
void CahnHilliardEquation<dim> :: run(
    const std::unordered_map<std::string, double> params,
    const double                                  totalSimTime
)
{
    this->setupSystem(params, totalSimTime);
    this->initializeValues();

    for(uint i = 0; i < 100; i++){
        this->timestep_number++;
        this->assembleSystem();
        this->solveSystem();
        this->outputResults();
    }
}

}


int main(){ 
    std::cout   << "Running" << std::endl << std::endl;

    std::unordered_map<std::string, double> params;

    params["eps"] = 1e-2;

    double totalSimTime = 10;

    cahnHilliard::CahnHilliardEquation<2> cahnHilliard;
    cahnHilliard.run(params, totalSimTime);

    std::cout << "Completed" << std::endl;

    return 0;
}
