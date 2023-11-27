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

template<int dim> 
InitialValuesC<dim>::InitialValuesC(double eps)
    : eps(eps)
{}
    
template<int dim> double InitialValuesC<dim> :: value(
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
    void reinitMandV();
    void initializeValues();
    void constructRightHandEta();
    void constructRightHandC();
    void solveC();
    void solveEta();
    void outputResults() const;

    Triangulation<dim>  triangulation;
    FE_Q<dim>           fe;
    DoFHandler<dim>     dof_handler;

    AffineConstraints<double>   constraints;

    SparsityPattern         sparsity_pattern;
    SparseMatrix<double>    mass_matrix;
    SparseMatrix<double>    laplace_matrix;
    SparseMatrix<double>    c_matrix;
    SparseMatrix<double>    eta_matrix;
    SparseMatrix<double>    newton_matrix;

    Vector<double>  c_solution;
    Vector<double>  c_old_solution;
    Vector<double>  eta_solution;
    Vector<double>  c_rhs;
    Vector<double>  eta_rhs;
    
    FullMatrix<double>  cell_newton_matrix;
    Vector<double>      cell_rhs_c;
    Vector<double>      cell_rhs_eta;

    double          timeStep;
    double          time;
    unsigned int    timestep_number;
    double          totalSimTime;

    double eps;
};

template<int dim> 
CahnHilliardEquation<dim> :: CahnHilliardEquation()
        : fe(1)
        , dof_handler(triangulation)
        , timeStep(1. / 256.)
        , time(timeStep)
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
    constraints.close();

    std::cout   << "Building sparsity pattern..."
                << std::endl;

    DynamicSparsityPattern dsp(
        dof_handler.n_dofs(),
        dof_handler.n_dofs()
    );
    DoFTools::make_sparsity_pattern(dof_handler,
                                    dsp,
                                    this->constraints);
    sparsity_pattern.copy_from(dsp);
    sparsity_pattern.compress();
}

template<int dim>
void CahnHilliardEquation<dim> :: reinitMandV()
{
    std::cout   << "Reinitializing matices based on new pattern..."
                << std::endl;

    mass_matrix.reinit(sparsity_pattern);
    laplace_matrix.reinit(sparsity_pattern);
    c_matrix.reinit(sparsity_pattern);
    eta_matrix.reinit(sparsity_pattern);

    std::cout   << "Reinitializing mass and laplace matrix..."
                << std::endl;

    MatrixCreator::create_mass_matrix(
        dof_handler,
        QGauss<dim>(fe.degree+1),
        mass_matrix
    );

    MatrixCreator::create_laplace_matrix(
        dof_handler,
        QGauss<dim>(fe.degree+1),
        laplace_matrix
    );

    this->newton_matrix.reinit(this->sparsity_pattern);

    std::cout   << "Reinitializing vectors..."
                << std::endl;

    this->c_solution.reinit(dof_handler.n_dofs());
    this->c_old_solution.reinit(dof_handler.n_dofs());
    this->eta_solution.reinit(dof_handler.n_dofs());

    this->c_rhs.reinit(dof_handler.n_dofs());
    this->eta_rhs.reinit(dof_handler.n_dofs());

    // Initialize memory for cell values and RHS
    this->cell_rhs_c.reinit(fe.n_dofs_per_cell());
    this->cell_rhs_eta.reinit(fe.n_dofs_per_cell());
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
    this->reinitMandV();

}

template<int dim>
void CahnHilliardEquation<dim> :: initializeValues()
{   
   
    std::cout   << "Initializing values for C" << std::endl;

    VectorTools::project(this->dof_handler,
                         this->constraints,
                         QGauss<dim>(fe.degree + 1),
                         InitialValuesC<dim>(this->eps),
                         this->c_old_solution);
    this->constraints.distribute(this->oldSolutionC);

    auto c_range = std::minmax_element(c_old_solution.begin(),
                                       c_old_solution.end());

    std::cout   << "Initial values propagated:\n"
                << "    Range: (" 
                    << *c_range.first << ", " 
                    << *c_range.second
                << ")" 
                << std::endl;

}

template<int dim>
void CahnHilliardEquation<dim> :: solveEta()
{   
    SolverControl               solverControl(
                                    1000,
                                    1e-8 * eta_rhs.l2_norm()
                                );
    SolverCG<Vector<double>>    cg(solverControl);

    this->constraints.condense(this->eta_matrix,
                               this->eta_rhs);
    PreconditionSSOR<SparseMatrix<double>> preconditioner;
    preconditioner.initialize(this->eta_matrix, 1.2);
    cg.solve(
        this->eta_matrix,
        this->eta_solution,
        this->eta_rhs,
        preconditioner
    );
    this->constraints.distribute(this->solutionEta);

    std::cout   << "    Eta solved: "
                << solverControl.last_step()
                << " CG iterations."
                << std::endl;
}

template<int dim>
void CahnHilliardEquation<dim> :: solveC()
{   
    SolverControl               solverControl(
                                    1000,
                                    1e-8 * c_rhs.l2_norm()
                                );
    SolverCG<Vector<double>>    cg(solverControl);

    this->constraints.condense(this->c_matrix,
                               this->c_rhs);
    PreconditionSSOR<SparseMatrix<double>> preconditioner;
    preconditioner.initialize(this->c_matrix, 1.2);
    cg.solve(
        this->c_matrix,
        this->c_solution,
        this->c_rhs,
        preconditioner
    );
    this->constraints.distribute(this->solutionC);

    std::cout   << "    C solved: "
                << solverControl.last_step()
                << " CG iterations."
                << std::endl;
}
    
template<int dim>
void CahnHilliardEquation<dim> :: outputResults() const
{

    DataOut<dim> dataOut;

    dataOut.attach_dof_handler(this->dof_handler);
    dataOut.add_data_vector(this->c_solution, "C");
    dataOut.add_data_vector(this->eta_solution, "Eta");
    dataOut.build_patches();

    const std::string filename = ("data/solution-" 
                                 + std::to_string(this->timestep_number) 
                                 + ".vtu");

    DataOutBase::VtkFlags vtk_flags;
    vtk_flags.compression_level = DataOutBase::VtkFlags::ZlibCompressionLevel::best_speed;
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
