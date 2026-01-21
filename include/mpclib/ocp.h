/**
 * @file ocp.h
 * @author Lena
 * @brief Wrapper class of the Optimal Control Problem Quadratic Programming (OCP-QP) solver using HPIPM.
 *
 * @details
 * This file defines the structures and methods for solving optimal control problems
 * using the HPIPM library. It includes the definition of OCP parameters, the OCPQP
 * class for managing the problem setup, and methods for relinearization, setting
 * targets, and solving the problem.
 */

#pragma once

#include "models/base_model.h"
#include "utils.h"
#include "hpipm/hpipm_s_ocp_qp.h"
#include "hpipm/hpipm_s_ocp_qp_dim.h"
#include "hpipm/hpipm_s_ocp_qp_sol.h"
#include "hpipm/hpipm_s_ocp_qp_ipm.h"

namespace mpclib {

/**
 * @brief Structure holding configuration parameters for the OCP solver.
 * @note
 * All matricies are expected to be in the correct shape. This means that 
 * all state cost (Q) matricies are of size (state_size, state_size),
 * all action cost (R) matricies are of size (action_size, action_size), 
 * and all state-action correlation (S) matricies are of size (state_size, action_size).
 */
struct OCPParams {
    int N; ///< Number of time steps in the OCP problem. This means there are N actions & N states to predict excluding the initial state.

    Mat Q1; ///< State cost for timestep 1 (\f$ x_1 \f$).
    Mat Q;  ///< State cost for intermediate timesteps 2 … N‑1.
    Mat Qf; ///< State cost for timestep N (\f$ x_N \f$)

    Mat R0; ///< Action cost for timestep 0 (\f$ u_0 \f$).
    Mat R;  ///< Action cost for intermediate timesteps 1 … N‑2.
    Mat Rf; ///< Action cost for timestep (\f$ u_{N-1} \f$).

    /**
     * @brief Enum defining warm start levels.
     */
    enum class WarmStartLevel {
        NONE = 0, ///< No warm start. Both state and input are "guessed" to be zero.
        STATE = 1, ///< Only the state is warm started. The input is "guessed" to be zero.
        STATE_AND_INPUT = 2, ///< Both state and input are warm started, i.e. reused from the previous solve.
        FULL = 3 ///< Full warm start, including state, input, other IPM variables.
    } warm_start_level = WarmStartLevel::STATE; ///< Default – reuse state only.

    int iterations = 5; ///< Maximum number of iterations per solve call
};

/**
 * @brief Class wrapping the OCP-QP solver.
 * @details
 * Handles initialization and solution of OCP problems using HPIPM. Provides functions to set targets,
 * initial conditions, constraints, and cost functions.
 */
struct OCPQP {
    const OCPParams& ocp_params; ///< Reference to OCP configuration.
    const Model& model; ///< Reference to the model defining system dynamics and constraints.

    s_ocp_qp_dim dim; ///< HPIPM dimension object.
    void* dim_mem; ///< Memory for HPIPM dimension object.

    s_ocp_qp qp; ///< HPIPM QP object.
    void* qp_mem; ///< Memory for HPIPM QP object.

    s_ocp_qp_sol qp_sol; ///< HPIPM solution object.
    void* qp_sol_mem; ///< Memory for HPIPM solution object.

    s_ocp_qp_ipm_ws workspace; ///< HPIPM workspace object.
    void* ipm_mem; ///< Memory for HPIPM workspace.

    s_ocp_qp_ipm_arg ipm_arg; ///< HPIPM solver argument object.
    void* ipm_arg_mem; ///< Memory for HPIPM solver argument object.

    /**
     * @brief Construct a partially initialized QP wrapper.
     *
     * @details
     * All HPIPM structures are allocated and the *static* parts of the problem
     * (dimensions, box constraints, state or action independent cost weights) are set up.
     * No linearization or initial state setting is performed here – call one 
     * of the `relinearize()` overloads and set the initial state before solving.
     *
     * @param model Dynamics model
     * @param ocp_params  Horizon length, cost weights and solver options.
     */
    OCPQP(const Model& model, const OCPParams& ocp_params);

    /**
     * @brief Destroy the OCPQP object
     * 
     * @details
     * Deallocates all HPIPM structures and memory.
     */
    ~OCPQP();

    /**
     * @brief Constrain the current state
     * 
     * @details
     * In HPIPM, the initial state is set as both the lower and upper bounds 
     * of the state variable. This constrains all future states, which are 
     * states in timesteps from \f$ [1, N] \f$ to evolve from the initial state.
     * 
     * @param x Initial state vector.
     */
    void set_initial_state(const Vec& x);

    /**
     * @brief Relinearize the model dynamics using a single state and input.
     * 
     * @details
     * This function will compute the jacobian of the model dynamics
     * and set the linearized dynamics in the QP problem. This is used to
     * approximate nonlinear models with a linear one - specifically, a
     * first order taylor expansion, where 
     * \f$ f(x, u) \approx J_{x}x + J_{u}u + [f(x_{nom}, u_{nom}) - J_{x}x_{nom} - J_{u}u_{nom}] \f$ 
     * 
     * @param x State vector to linearize around.
     * @param u Action vector to linearize around.
     * @param first_stage_dt_override Override the time step for the first stage. If negative, the default time step from the model parameters is used.
     * 
     * @see @ref mpclib::Model::autodiff
     */
    void relinearize(
        const Vec& x, 
        const Vec& u, 
        std::optional<float> first_stage_dt_override = std::nullopt);

    /**
     * @brief Relinearize the model dynamics by evolving a state from a sequence of actions
     * 
     * @details 
     * Similar to @ref relinearize, but this function will evolve the state
     * using the model dynamics @ref mpclib::Model::infer and the sequence of actions. This essentially "predicts"
     * the future states of the model, and moves the operating points to the predicted states each state.
     * This prevents errors  from linearization in highly non-linear models, when \f$ x \f$ is far from \f$ x_{nom} \f$,
     * or when \f$ u \f$ is far from \f$ u_{nom} \f$.
     * 
     * @param x the initial state to evolve from.
     * @param u list of actions to evolve the state with.
     * @param first_stage_dt_override Override the time step for the first stage. If negative, the default time step from the model parameters is used.
     *
     * @see @ref mpclib::Model::autodiff
     * @note If the size of `u` is less than `N`, the last state & action will be used to fill the rest of the dynamics matricies.
     */
    void relinearize(
        Vec x, 
        const std::vector<Vec>& u, 
        std::optional<float> first_stage_dt_override = std::nullopt);

    /**
     * @brief Relinearize the model dynamics from a sequence of states and actions
     * 
     * @details
     * Similar to @ref relinearize, but this function will set the dynamics matricies at each timestep
     * to the ones computed from the state and action pairs. Note that this differs from 
     * @ref relinearize(Vec x, const std::vector<Vec>& u) in that it does not evolve any states.
     * 
     * @param x States to linearize around at each timestep.
     * @param u Actions to linearize around at each timestep.
     * @param first_stage_dt_override Override the time step for the first stage. If negative, the default time step from the model parameters is used.
     *
     * @see @ref mpclib::Model::autodiff
     * @note If the size of `u` or `x` is less than `N`, the last state & action will be used to fill the rest of the dynamics matricies.
     * Additionally, if the size of `x` differs from `u`, the smaller of the two will be used.
     */
    void relinearize(
        const std::vector<Vec>& x, 
        const std::vector<Vec>& u, 
        std::optional<float> first_stage_dt_override = std::nullopt);

    /**
     * @brief Set a single target state across all N timesteps
     * 
     * @param x_desired Desired state vector to achieve
     */
    void set_target_state(const Vec& x_desired);

    /**
     * @brief Set possibly unique target states in each timestep
     * 
     * @details
     * At each timestep \f$ i \f$ in the range \f$ [1, N] \f$, the target state is set to the \f$ i-1 \f$ element in the list.
     *
     * @param x_desired List of desired states to achieve at each timestep
     */
    void set_target_state(const std::vector<Vec>& x_desired);

    /**
     * @brief Set a single target input across all N timesteps
     * 
     * @param u_desired Desired action vector to achieve
     */
    void set_target_input(const Vec& u_desired);

    /**
     * @brief Set possibly unique target inputs in each timestep
     * 
     * @param u_desired 
     */
    void set_target_input(const std::vector<Vec>& u_desired);

    /**
     * @brief Solves the OCP problem using HPIPM.
     * 
     * @param silent Whether to suppress output to `cout` upon an error from the solver
     *
     * @return int HPIPM solver status code
     * @see https://github.com/giaf/hpipm/blob/master/include/hpipm_common.h for return code meanings
     */
    int solve(bool silent = true);

    /**
     * @brief Get all (timesteps \f$ [1, N] \f$) solution states from the @ref qp_sol
     * 
     * @return std::vector<Vec> State vectors of the solution at each timestep. The i-th element corresponds to the state at timestep i+1.
     */
    std::vector<Vec> get_solution_states() const;

    /**
     * @brief Get all (timesteps \f$ [0, N-1] \f$) solution actions from the @ref qp_sol
     * 
     * @return std::vector<Vec> Action vectors of the solution at each timestep. The i-th element corresponds to the action at timestep i.
     */
    std::vector<Vec> get_solution_actions() const;

    /**
     * @brief Get the solution state at a specific timestep.
     *
     * @note States are 1-indexed in HPIPM, so the first (non-initial) state is given at timestep 1 and the last state is at timestep N.
     * 
     * @param i timestep index in the range \f$ [1, N] \f$.
     * @return Vec State vector at timestep i.
     */
    Vec get_solution_state(int i) const;

    /**
     * @brief Get the solution action object
     * 
     * @note Actions are 0-indexed in HPIPM, so the first action is 0 and the last action is N-1.
     * @param i 
     * @return Vec 
     */
    Vec get_solution_action(int i) const;

    /**
     * @brief "Push down" the solution state and actions by one timestep.
     * 
     * @details
     * This function helps hot start the solver by reusing the solution from the previous timestep.
     * When hot start solving, the previous solution was solved at a previous world time.
     * Thus, it makes sense that the solution at timestep \f$ i \f$ in current time should be close to
     * the solution at timestep \f$ i-1 \f$ in the previous time.
     * 
     * @note This function does not do anything to the state at timestep N, and the action at timestep N-1.
     */
    void push_down_solutions() { push_down_solution_states(); push_down_solution_actions(); }

    /**
     * @brief "Push down" the solution state by one timestep.
     * 
     * @details
     * This function helps hot start the solver by reusing the solution from the previous timestep.
     * When hot start solving, the previous solution was solved at a previous world time.
     * Thus, it makes sense that the solution at timestep \f$ i \f$ in current time should be close to
     * the solution at timestep \f$ i-1 \f$ in the previous time.
     * 
     * @note This function does not do anything to the state at timestep N
     */
    void push_down_solution_states();

    /**
     * @brief "Push down" the solution action by one timestep.
     *      
     * @details
     * This function helps hot start the solver by reusing the solution from the previous timestep.
     * When hot start solving, the previous solution was solved at a previous world time.
     * Thus, it makes sense that the solution at timestep \f$ i \f$ in current time should be close to
     * the solution at timestep \f$ i-1 \f$ in the previous time.
     * 
     * @note This function does not do anything to the action at timestep N-1
     */
    void push_down_solution_actions();

private:
    /**
    * @brief Uses @ref ocp_params to input dimensions of the problem into HPIPM.
    */
    void setup_dimensions();

    /**
     * @brief Uses @ref ocp_params input box & general constraints into HPIPM.
     */
    void setup_constraints();

    /**
     * @brief Uses @ref ocp_params to input cost matricies at each timestep into HPIPM.
     */
    void setup_costs();
};
}