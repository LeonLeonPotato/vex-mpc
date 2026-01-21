/**
 * @file base_model.h
 * @author Leon
 * @brief Base model class for the library
 * @date 2025-04-20
 */

#pragma once

#include "mpclib/utils.h"

namespace mpclib {

/**
 * @brief Represents a lower bound & upper bound constraint on a variable.
 * 
 * @details
 * This structure is used to define constraints on the state and action variables
 * in the model. Each constraint has a lower and upper bound, as well as an index
 * that indicates which variable it applies to. The index is used to map the constraint
 * to the corresponding variable in the state or action vector.
 * 
 * The constraints are used in the optimization problem to ensure that the solution
 * respects the physical limits of the system being modeled. For example, a constraint
 * on the velocity of a robot would ensure that the robot does not exceed its maximum speed.
 */
struct Constraint {
    float lower_bound; ///< Lower bound of the target variable - the variable cannot be lower than this value.
    float upper_bound; ///< Upper bound of the target variable - the variable cannot be higher than this value.
    int index; // < Index of the target variable, e.g. an index of 1 means the second variable in the state or action vector.
};

/**
 * @brief Abstract base class for dynamic models used in model predictive control (MPC).
 *
 * @details
 * This class defines the common interface for all dynamic models used in the MPC library.
 * Derived classes must define the dimensionality of the state and action vectors,
 * as well as the dynamics functions for simulation and automatic differentiation.
 *
 * The model also supports various types of constraints:
 * - **State constraints**: Variable-wise lower and upper bounds on the state vector.
 * - **Action constraints**: Variable-wise lower and upper bounds on the action vector.
 * - **General constraints**: Arbitrary linear constraints on the state and/or action vectors.
 */
class Model {
public: 
    /**
     * @struct BaseParams
     * @brief Represents the base parameters for a model.
     *
     * This structure contains the fundamental configuration parameters
     * required for a model, such as the time step.
     */
    struct BaseParams {
        float dt; ///< Time step in seconds. This defines the interval at which the model operates or updates.
    };

    /**
     * @brief Returns the number of states variables in the model.
     * 
     * @return `constexpr int` The size of the state vector
     */
    virtual constexpr int state_size() const = 0;

    /**
     * @brief Returns the number of action variables in the model.
     * 
     * @return `constexpr int` The size of the action vector
     */
    virtual constexpr int action_size() const = 0;

    /**
     * @brief Get the model parameters
     * 
     * @return `const BaseParams&` - constant reference to the model parameters
     */
    virtual const BaseParams& get_params() const = 0;

    /**
     * @brief Predict the next state using the model dynamics, with automatic differentiation support
     * 
     * @details
     * This function is used to compute the jacobian of the model dynamics using autodiff.
     * 
     * @param x Input state
     * @param u Input action
     * @param dt_override Override the time step for this prediction. If negative, the default time step from the model parameters is used.
     * @return ADVec - Predicted next state with the model dynamics given the input and action
     */
    virtual ADVec autodiff(const ADVec& x, const ADVec& u, double dt_override) const = 0;

    /**
     * @brief Predict the next state using the model dynamics
     * 
     * @param x Input state
     * @param u Input action
     * @param dt_override Override the time step for this prediction. If negative, the default time step from the model parameters is used.
     * @return Vec - Predicted next state with the model dynamics given the input and action
     */
    virtual Vec infer(const Vec& x, const Vec& u, float dt_override) const = 0;

    /**
     * @brief Get the state constraints
     * 
     * @return const std::vector<Constraint>& constant reference to the list of state constraints at every timestep
     * @see @ref mpclib::Constraint
     */
    virtual const std::vector<Constraint>& state_constraints() const { return state_constraints_; }

    /**
     * @brief Get the action constraints
     * 
     * @return const std::vector<Constraint>& - constant reference to the list of action constraints at every timestep
     * @see @ref mpclib::Constraint
     */
    virtual const std::vector<Constraint>& action_constraints() const { return action_constraints_; }

    /**
     * @brief Get the general constraints
     * 
     * @return const std::vector<Constraint>& - constant reference to the list of general constraints at every timestep
     * @see @ref mpclib::Constraint
     */
    virtual const std::vector<Constraint>& general_constraints() const { return general_constraints_; }

    /**
     * @brief Get the general state constraints matrix
     * 
     * @details 
     * The general state constraints matrix is a matrix of dimensions \f$ [\mathrm{Number\,of\,general\,constraints}, \mathrm{state\,size}] \f$.
     * Each row of the matrix corresponds to a general constraint, and each column corresponds to a state variable.
     * In HPIPM, this matrix is multiplied by the state vector at every timestep.
     *
     * @return `const Mat&` - constant reference to the general state constraints matrix
     */
    virtual const Mat& general_constraints_state_matrix() const { return general_state_matrix; }

    /**
     * @brief Get the general action constraints matrix
     * 
     * @details 
     * The general action constraints matrix is a matrix of dimensions \f$ [\mathrm{Number\,of\,general\,constraints}, \mathrm{action\,size}] \f$.
     * Each row of the matrix corresponds to a general constraint, and each column corresponds to an action variable.
     * In HPIPM, this matrix is multiplied by the action vector at every timestep.
     *
     * @return `const Mat&` - constant reference to the general action constraints matrix
     */
    virtual const Mat& general_constraints_action_matrix() const { return general_action_matrix; }

protected:
    std::vector<Constraint> state_constraints_; ///< List of state constraints at every timestep
    std::vector<Constraint> action_constraints_; ///< List of action constraints at every timestep
    std::vector<Constraint> general_constraints_; ///< List of general constraints at every timestep

    Mat general_state_matrix; ///< General state constraint matrix, of dimensions \f$ [\mathrm{Number\,of\,general\,constraints}, \mathrm{state\,size}] \f$
    Mat general_action_matrix; ///< General action constraint matrix, of dimensions \f$ [\mathrm{Number\,of\,general\,constraints}, \mathrm{action\,size}] \f$
};
}