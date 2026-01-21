/**
 * @file ocp.cpp
 * @author Lena
 * @brief Wrapper class of the Optimal Control Problem Quadratic Programming (OCP-QP) solver using HPIPM.
 * 
 * This file contains the implementation of the OCPQP class, which is responsible for setting up and solving
 * optimal control problems using the HPIPM library. The implementation includes methods for setting up problem
 * dimensions, constraints, costs, and relinearizing the dynamics. It also provides functionality for setting
 * target states and solving the problem.
 * 
 * @details
 * The OCPQP class leverages the HPIPM library to solve quadratic programming problems that arise in optimal
 * control. The implementation includes helper functions for creating masks, indices, and bounds from constraints,
 * as well as methods for relinearizing the system dynamics using automatic differentiation. The solver supports
 * warm-starting and allows for customization of solver parameters such as iteration limits and accuracy levels.
 * 
 * Key features:
 * - Setup of problem dimensions, constraints, and costs.
 * - Support for state, action, and general constraints.
 * - Automatic differentiation for relinearization of dynamics.
 * - Warm-starting and customizable solver parameters.
 * - Debugging utilities for printing arrays and intermediate results.
 * 
 * Dependencies:
 * - HPIPM library for solving quadratic programming problems.
 * - Eigen library for matrix and vector operations.
 * - Autodiff library for automatic differentiation of nonlinear functions.
 * 
 * Usage:
 * - Instantiate the OCPQP class with a model and OCP parameters.
 * - Use the provided methods to set up the problem, relinearize dynamics, and solve the problem.
 * - Retrieve the solution and status using the appropriate methods.
 * 
 * @author Leon
 * @date 2025-4-20
 */

#include "ocp.h"
#include "hpipm/hpipm_s_ocp_qp_ipm.h"
#include <algorithm>
#include <stdexcept>
#include <vector>

using namespace mpclib;

/**
 * @brief Create an HPIPM bit mask from a list of constraints object
 *
 * @details  
 * Loops over the constraints, setting the 'index' parameter of each 
 * constraint to 1, and everything else to 0. 
 * In HPIPM terms, this activates the constraint at that index.
 * 
 * @param arr pointer to the C-style array to write to
 * @param size size of the constraints vector and the array to copy to
 * @param constraints vector of constraints
 */
static void create_mask_from_constraints(float* arr, size_t size, const std::vector<Constraint>& constraints) {
    if (arr == nullptr) return;
    std::fill(arr, arr + size, 0.0f);
    for (auto& c : constraints) {
        arr[c.index] = 1.0f;
    }
}

/**
 * @brief Create an HPIPM index from a list of constraints object
 *
 * @details 
 * Copies all 'index' parameters of every constraint into a C-style 
 * array of the same size. In HPIPM terms, this tells the solver 
 * the index of every variable that is targetted in the constraint.
 * 
 * @param arr pointer to the C-style array to copy to
 * @param size size of the constraints vector and the array to copy to
 * @param constraints vector of constraints
 */
static void create_index_from_constraints(int* arr, size_t size, const std::vector<Constraint>& constraints) {
    if (arr == nullptr) return;
    for (int i = 0; i < size; ++i) {
        arr[i] = constraints[i].index;
    }
}

/**
 * @brief Create lower and upper bounds from a list of constraints object
 * 
 * @details 

 * 
 * @param lower pointer to the lower bound array to copy to
 * @param upper pointer to the upper bound array to copy to
 * @param constraints vector of constraints
 */
static void create_bounds_from_constraints(float* lower, float* upper, const std::vector<Constraint>& constraints) {
    if (lower == nullptr || upper == nullptr) return;
    for (int i = 0; i < constraints.size(); ++i) {
        lower[i] = constraints[i].lower_bound;
        upper[i] = constraints[i].upper_bound;
    }
}

/**
 * @brief Prints an array to stdout
 * @note Used for debugging
 * 
 * @tparam T the type of every element in the array
 * @param arr pointer to the array to print from 
 * @param size size of the array
 */
template <typename T>
static void print_arr(T* arr, size_t size) {
    std::cout << "[";
    for (int i = 0; i < size; ++i) {
        std::cout << arr[i];
        if (i != size - 1) std::cout << ", ";
    }
    std::cout << "]\n";
}

/**
 * @brief static method to wrap the autodiff function for the model
 * 
 * @param model Model object to call the autodiff function on
 * @param x state operating vector to linearize around
 * @param u action operating vector to linearize around
 * @return auto 
 */
static auto autodiff_model_wrapper(const Model& model, const ADVec& x, const ADVec& u, double dt_override) {
    return model.autodiff(x, u, dt_override);
}

OCPQP::OCPQP(const Model& model, const OCPParams& ocp_params)
    : model(model), ocp_params(ocp_params) 
{
    // Allocation of dimension memory
    hpipm_size_t memsize = s_ocp_qp_dim_memsize(ocp_params.N);
    dim_mem = malloc(memsize);
    s_ocp_qp_dim_create(ocp_params.N, &dim, dim_mem);

    setup_dimensions();

    // Allocate internal HPIPM memory for solver itself
    hpipm_size_t qp_size = s_ocp_qp_memsize(&dim);
    qp_mem = malloc(qp_size);
    s_ocp_qp_create(&dim, &qp, qp_mem);

    setup_constraints();
    setup_costs();

    // Additional solver setup (solution, workspace, etc)
    hpipm_size_t qp_sol_size = s_ocp_qp_sol_memsize(&dim);
    qp_sol_mem = malloc(qp_sol_size);
	s_ocp_qp_sol_create(&dim, &qp_sol, qp_sol_mem);

	hpipm_size_t ipm_arg_size = s_ocp_qp_ipm_arg_memsize(&dim);
	ipm_arg_mem = malloc(ipm_arg_size);
	s_ocp_qp_ipm_arg_create(&dim, &ipm_arg, ipm_arg_mem);
	s_ocp_qp_ipm_arg_set_default(hpipm_mode::SPEED, &ipm_arg); // In this case accuracy is not as important as speed
    int warm_start_level = static_cast<int>(ocp_params.warm_start_level);
    s_ocp_qp_ipm_arg_set_warm_start(&warm_start_level, &ipm_arg);
    int it = ocp_params.iterations;
    s_ocp_qp_ipm_arg_set_iter_max(&it, &ipm_arg);

    hpipm_size_t ipm_size = s_ocp_qp_ipm_ws_memsize(&dim, &ipm_arg);
	ipm_mem = malloc(ipm_size);
	s_ocp_qp_ipm_ws_create(&dim, &ipm_arg, &workspace, ipm_mem);
}

OCPQP::~OCPQP() {
    free(dim_mem);
    free(qp_mem);
    free(qp_sol_mem);
    free(ipm_arg_mem);
    free(ipm_mem);
}

void OCPQP::setup_dimensions() {
    // Set the dimensions of the problem
    int state_space[ocp_params.N + 1];
    std::fill_n(state_space, ocp_params.N + 1, model.state_size());

    int action_space[ocp_params.N + 1];
    std::fill_n(action_space, ocp_params.N + 1, model.action_size());
    action_space[ocp_params.N] = 0; // no action at the last timestep as we do not penalize the next state

    int state_bounds[ocp_params.N + 1];
    std::fill_n(state_bounds, ocp_params.N + 1, model.state_constraints().size());
    state_bounds[0] = model.state_size(); // initial state constraint

    int action_bounds[ocp_params.N + 1];
    std::fill_n(action_bounds, ocp_params.N + 1, model.action_constraints().size());
    action_bounds[ocp_params.N] = 0; // no action box constraints at the last timestep

    int general_bounds[ocp_params.N + 1];
    std::fill_n(general_bounds, ocp_params.N + 1, model.general_constraints().size());
    general_bounds[ocp_params.N] = 0; // no general constraints at the last timestep (since no action at last timestep)

    // TODO: implement soft constraining in model
    int soft_bounds[ocp_params.N + 1];
    std::fill_n(soft_bounds, ocp_params.N + 1, 0);

    // Pass into HPIPM
    s_ocp_qp_dim_set_all(state_space, action_space,
        state_bounds, action_bounds, general_bounds,
        soft_bounds, soft_bounds, soft_bounds,
        &dim);

    // Set equality
    s_ocp_qp_dim_set_nbxe(0, model.state_size(), &dim);
}

void OCPQP::setup_constraints() {
    // Create C style constraint masks & indicies from the model
    float state_bound_mask[model.state_size()];
    create_mask_from_constraints(state_bound_mask, model.state_size(), model.state_constraints());

    float action_bound_mask[model.action_size()];
    create_mask_from_constraints(action_bound_mask, model.action_size(), model.action_constraints());

    for(int i = 0; i < ocp_params.N; i++) {
        // State constraints vary from [1, N]
        s_ocp_qp_set_lbx_mask(i+1, state_bound_mask, &qp);
        s_ocp_qp_set_ubx_mask(i+1, state_bound_mask, &qp);

        // Action constraints vary from [0, N-1]
        s_ocp_qp_set_lbu_mask(i, action_bound_mask, &qp);
        s_ocp_qp_set_ubu_mask(i, action_bound_mask, &qp);
    }

    // Avoid UB when creating the index since it could result in zero-sized arrays (I LOVE CPP!!!!)
    if (model.state_constraints().size() > 0) {
        int state_bound_index[model.state_constraints().size()];
        create_index_from_constraints(state_bound_index, model.state_constraints().size(), model.state_constraints());

        float lower_bound[model.state_constraints().size()];
        float upper_bound[model.state_constraints().size()];
        create_bounds_from_constraints(lower_bound, upper_bound, model.state_constraints());

        // [1, N]
        for(int i = 1; i <= ocp_params.N; i++) {
            s_ocp_qp_set_idxbx(i, state_bound_index, &qp);

            s_ocp_qp_set_lbx(i, lower_bound, &qp);
            s_ocp_qp_set_ubx(i, upper_bound, &qp);
        }
    }

    if (model.action_constraints().size() > 0) {
        int action_bound_index[model.action_constraints().size()];
        create_index_from_constraints(action_bound_index, model.action_constraints().size(), model.action_constraints());

        float lower_bound[model.action_constraints().size()];
        float upper_bound[model.action_constraints().size()];
        create_bounds_from_constraints(lower_bound, upper_bound, model.action_constraints());

        // [0, N-1]
        for(int i = 0; i < ocp_params.N; i++) {
            s_ocp_qp_set_idxbu(i, action_bound_index, &qp);

            s_ocp_qp_set_lbu(i, lower_bound, &qp);
            s_ocp_qp_set_ubu(i, upper_bound, &qp);
        }
    }

    if (model.general_constraints().size()) {
        float general_bound_mask[model.general_constraints().size()];
        create_mask_from_constraints(general_bound_mask, model.general_constraints().size(), model.general_constraints());

        float lower_bound[model.general_constraints().size()];
        float upper_bound[model.general_constraints().size()];
        create_bounds_from_constraints(lower_bound, upper_bound, model.general_constraints());

        // General constraints also vary from [0, N-1] (As it depends on action)
        for (int i = 0; i < ocp_params.N; i++) {
            s_ocp_qp_set_lg_mask(i, general_bound_mask, &qp);
            s_ocp_qp_set_ug_mask(i, general_bound_mask, &qp);

            s_ocp_qp_set_C(i, const_cast<float*>(model.general_constraints_state_matrix().data()), &qp);
            s_ocp_qp_set_D(i, const_cast<float*>(model.general_constraints_action_matrix().data()), &qp);

            s_ocp_qp_set_lg(i, lower_bound, &qp);
            s_ocp_qp_set_ug(i, upper_bound, &qp);
        }
    }

    // State mask and index to constrain the initial state (timestep=0) to be the current state
    float initial_state_constraint_mask[model.state_size()];
    int initial_state_constraint_index[model.state_size()];
    std::fill_n(initial_state_constraint_mask, model.state_size(), 1.0f);
    std::iota(initial_state_constraint_index, initial_state_constraint_index + model.state_size(), 0);
    s_ocp_qp_set_lbx_mask(0, initial_state_constraint_mask, &qp);
    s_ocp_qp_set_ubx_mask(0, initial_state_constraint_mask, &qp);
    s_ocp_qp_set_idxbx(0, initial_state_constraint_index, &qp);
    s_ocp_qp_set_idxbxe(0, initial_state_constraint_index, &qp);
}

void OCPQP::setup_costs() {
    Mat zeros = Mat::Zero(model.state_size(), model.state_size());

    // Penalize every state from [1, N-1]
    for (int i = 1; i < ocp_params.N; i++) {
        s_ocp_qp_set_Q(i, const_cast<float*>((ocp_params.Q).eval().data()), &qp);
        s_ocp_qp_set_R(i-1, const_cast<float*>(ocp_params.R.data()), &qp);
        s_ocp_qp_set_S(i, zeros.data(), &qp); // No action-state correlation cost
        s_ocp_qp_set_q(i, zeros.data(), &qp); // No linear state cost (Implicitly makes the optimal state equal to the zero vector)
        s_ocp_qp_set_r(i, zeros.data(), &qp); // No linear action cost
    }

    // Final state penalty
    s_ocp_qp_set_Q(ocp_params.N, const_cast<float*>(ocp_params.Qf.data()), &qp);
    s_ocp_qp_set_R(ocp_params.N-1, const_cast<float*>(ocp_params.Rf.data()), &qp);
    s_ocp_qp_set_S(ocp_params.N, zeros.data(), &qp);
    s_ocp_qp_set_q(ocp_params.N, zeros.data(), &qp);
    s_ocp_qp_set_r(ocp_params.N, zeros.data(), &qp);

    // Initial state penalty
    s_ocp_qp_set_Q(1, const_cast<float*>(ocp_params.Q1.data()), &qp);
    s_ocp_qp_set_R(0, const_cast<float*>(ocp_params.R0.data()), &qp);
}

void OCPQP::set_initial_state(const Vec& x) {
    // Its okay (I think) to use const_cast here because the HPIPM API is not const-correct
    s_ocp_qp_set_lbx(0, const_cast<float*>(x.data()), &qp);
    s_ocp_qp_set_ubx(0, const_cast<float*>(x.data()), &qp);
}

void OCPQP::relinearize(const Vec& x, const Vec& u, std::optional<float> first_stage_dt_override) {
    float dt = first_stage_dt_override.value_or(model.get_params().dt);
    // First, we linearize with dt override
    ADVec fout;
    ADVec ad_x = x.cast<autodiff::real>();
    ADVec ad_u = u.cast<autodiff::real>();
    Mat jx = autodiff::jacobian(autodiff_model_wrapper, autodiff::wrt(ad_x), autodiff::at(model, ad_x, ad_u, (double) dt), fout).cast<float>();
    Mat ju = autodiff::jacobian(autodiff_model_wrapper, autodiff::wrt(ad_u), autodiff::at(model, ad_x, ad_u, (double) dt), fout).cast<float>();
    Vec c = fout.cast<float>() - (jx * x) - (ju * u);

    // set t = 0 dynamics here
    s_ocp_qp_set_A(0, jx.data(), &qp);
    s_ocp_qp_set_B(0, ju.data(), &qp);
    s_ocp_qp_set_b(0, c.data(), &qp);

    // Then, we linearize with the default dt
    ad_x = x.cast<autodiff::real>();
    ad_u = u.cast<autodiff::real>();
    jx = autodiff::jacobian(autodiff_model_wrapper, autodiff::wrt(ad_x), autodiff::at(model, ad_x, ad_u, model.get_params().dt), fout).cast<float>();
    ju = autodiff::jacobian(autodiff_model_wrapper, autodiff::wrt(ad_u), autodiff::at(model, ad_x, ad_u, model.get_params().dt), fout).cast<float>();
    c = fout.cast<float>() - (jx * x) - (ju * u);

    // [1, N-1] because the last state (N) does not need to transition to the next (N+1) (It doesnt exist)
    for (int i = 1; i < ocp_params.N; i++) {
        s_ocp_qp_set_A(i, jx.data(), &qp);
        s_ocp_qp_set_B(i, ju.data(), &qp);
        s_ocp_qp_set_b(i, c.data(), &qp);
    }
}

void OCPQP::relinearize(Vec x, const std::vector<Vec>& u, std::optional<float> first_stage_dt_override) {
    ADVec ad_x = x.cast<autodiff::real>();
    Mat jx, ju;
    Vec c;
    for (int i = 0; i < u.size(); i++) {
        ADVec fout;
        ADVec ad_u = u[i].cast<autodiff::real>();
        double dt = i == 0 ? first_stage_dt_override.value_or(model.get_params().dt) : model.get_params().dt;
        jx = autodiff::jacobian(autodiff_model_wrapper, autodiff::wrt(ad_x), autodiff::at(model, ad_x, ad_u, dt), fout).cast<float>();
        ju = autodiff::jacobian(autodiff_model_wrapper, autodiff::wrt(ad_u), autodiff::at(model, ad_x, ad_u, dt), fout).cast<float>();
        c = fout.cast<float>() - (jx * x) - (ju * u[i]);

        s_ocp_qp_set_A(i, jx.data(), &qp);
        s_ocp_qp_set_B(i, ju.data(), &qp);
        s_ocp_qp_set_b(i, c.data(), &qp);

        ad_x = fout.cast<float>().cast<autodiff::real>();
    }

    for (int i = u.size(); i < ocp_params.N; i++) {
        s_ocp_qp_set_A(i, jx.data(), &qp);
        s_ocp_qp_set_B(i, ju.data(), &qp);
        s_ocp_qp_set_b(i, c.data(), &qp);
    }
}

void OCPQP::relinearize(const std::vector<Vec>& x, const std::vector<Vec>& u, std::optional<float> first_stage_dt_override) {
    Mat jx, ju;
    Vec c;

    for (int i = 0; i < std::min(x.size(), u.size()); i++) {
        ADVec fout;
        ADVec ad_x = x[i].cast<autodiff::real>();
        ADVec ad_u = u[i].cast<autodiff::real>();
        double dt = i == 0 ? first_stage_dt_override.value_or(model.get_params().dt) : model.get_params().dt;
        jx = autodiff::jacobian(autodiff_model_wrapper, autodiff::wrt(ad_x), autodiff::at(model, ad_x, ad_u, dt), fout).cast<float>();
        ju = autodiff::jacobian(autodiff_model_wrapper, autodiff::wrt(ad_u), autodiff::at(model, ad_x, ad_u, dt), fout).cast<float>();
        c = fout.cast<float>() - (jx * x[i]) - (ju * u[i]);

        s_ocp_qp_set_A(i, jx.data(), &qp);
        s_ocp_qp_set_B(i, ju.data(), &qp);
        s_ocp_qp_set_b(i, c.data(), &qp);
    }

    for (int i = u.size(); i < ocp_params.N; i++) {
        s_ocp_qp_set_A(i, jx.data(), &qp);
        s_ocp_qp_set_B(i, ju.data(), &qp);
        s_ocp_qp_set_b(i, c.data(), &qp);
    }
}

void OCPQP::set_target_state(const Vec& x_desired) {
    Vec q1_cost = -ocp_params.Q1 * x_desired;
    s_ocp_qp_set_q(1, q1_cost.data(), &qp);

    Vec q_cost = -ocp_params.Q * x_desired;
    for (int i = 2; i < ocp_params.N; i++) {
        s_ocp_qp_set_q(i, q_cost.data(), &qp);
    }

    Vec qf_cost = -ocp_params.Qf * x_desired;
    s_ocp_qp_set_q(ocp_params.N, qf_cost.data(), &qp);
}

void OCPQP::set_target_state(const std::vector<Vec>& x_desired) {
    if (x_desired.size() != ocp_params.N) {
        throw std::invalid_argument("x_desired size must match the number of timesteps (N)");
    }

    Vec q_cost(model.state_size());

    // Initial cost
    q_cost = -ocp_params.Q1 * x_desired[0];
    s_ocp_qp_set_q(1, q_cost.data(), &qp);

    // Intermediate costs
    for (int i = 2; i < ocp_params.N; i++) {
        q_cost = -ocp_params.Q * x_desired[i-1];
        s_ocp_qp_set_q(i, q_cost.data(), &qp);
    }

    // Final cost
    q_cost = -ocp_params.Qf * x_desired[ocp_params.N-1];
    s_ocp_qp_set_q(ocp_params.N, q_cost.data(), &qp);
}

void OCPQP::set_target_input(const Vec& u_desired) {
    Vec r0_cost = -ocp_params.R0 * u_desired;
    s_ocp_qp_set_r(0, r0_cost.data(), &qp);

    Vec r_cost = -ocp_params.R * u_desired;
    for (int i = 1; i < ocp_params.N-1; i++) {
        s_ocp_qp_set_r(i, r_cost.data(), &qp);
    }

    Vec rf_cost = -ocp_params.Rf * u_desired;
    s_ocp_qp_set_r(ocp_params.N-1, rf_cost.data(), &qp);
}

void OCPQP::set_target_input(const std::vector<Vec>& u_desired) {
    if (u_desired.size() != ocp_params.N) {
        throw std::invalid_argument("u_desired size must match the number of timesteps (N)");
    }

    Vec r0_cost = -ocp_params.R0 * u_desired[0];
    s_ocp_qp_set_r(0, r0_cost.data(), &qp);

    for (int i = 1; i < ocp_params.N-1; i++) {
        Vec r_cost = -ocp_params.R * u_desired[i-1];
        s_ocp_qp_set_r(i, r_cost.data(), &qp);
    }

    Vec rf_cost = -ocp_params.Rf * u_desired[ocp_params.N-1];
    s_ocp_qp_set_r(ocp_params.N-1, rf_cost.data(), &qp);
}


int OCPQP::solve(bool silent) {
    s_ocp_qp_ipm_solve(&qp, &qp_sol, &ipm_arg, &workspace);

    int status;
    s_ocp_qp_ipm_get_status(&this->workspace, &status);
    if (silent) return status;

    switch (status) {
        case hpipm_status::INCONS_EQ:
            printf("Solving error code %d: INCONS_EQ\n", status);
            break;
        case hpipm_status::NAN_SOL:
            printf("Solving error code %d: NAN_SOL\n", status);
            break;
        default:
            break;
    }
    // MAX_ITER and MIN_STEP are not errors, just warnings

    return status;
}

std::vector<Vec> OCPQP::get_solution_states() const {
    std::vector<Vec> states(ocp_params.N);
    for (int i = 0; i < ocp_params.N; i++) {
        states[i].resize(model.state_size());
        s_ocp_qp_sol_get_x(i+1, const_cast<s_ocp_qp_sol*>(&qp_sol), states[i].data()); // Const cast because HPIPM API is not const-correct
        if (states[i].hasNaN()) states[i].setZero();
    }
    return states;
}

std::vector<Vec> OCPQP::get_solution_actions() const {
    std::vector<Vec> actions(ocp_params.N);
    for (int i = 0; i < ocp_params.N; i++) {
        actions[i].resize(model.action_size());
        s_ocp_qp_sol_get_u(i, const_cast<s_ocp_qp_sol*>(&qp_sol), actions[i].data()); 
        if (actions[i].hasNaN()) actions[i].setZero();
    }
    return actions;
}

Vec OCPQP::get_solution_state(int i) const {
    Vec state(model.state_size());
    s_ocp_qp_sol_get_x(i+1, const_cast<s_ocp_qp_sol*>(&qp_sol), state.data()); 
    return state;
}

Vec OCPQP::get_solution_action(int i) const {
    Vec action(model.action_size());
    s_ocp_qp_sol_get_u(i, const_cast<s_ocp_qp_sol*>(&qp_sol), action.data()); 
    return action;
}

void OCPQP::push_down_solution_states() {
    Vec x_buffer(model.state_size());
    for (int i = 0; i < ocp_params.N - 1; i++) {
        s_ocp_qp_sol_get_x(i+1, const_cast<s_ocp_qp_sol*>(&qp_sol), x_buffer.data()); 
        s_ocp_qp_sol_set_x(i, x_buffer.data(), &qp_sol);
    }
}

void OCPQP::push_down_solution_actions() {
    Vec u_buffer(model.action_size());
    for (int i = 0; i < ocp_params.N - 1; i++) {
        s_ocp_qp_sol_get_u(i+1, const_cast<s_ocp_qp_sol*>(&qp_sol), u_buffer.data()); 
        s_ocp_qp_sol_set_u(i, u_buffer.data(), &qp_sol);
    }
}