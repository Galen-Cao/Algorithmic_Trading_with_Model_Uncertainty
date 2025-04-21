import numpy as np

# import pandas as pd
import matplotlib.pyplot as plt
from scipy.linalg import expm


class RobustAgent:
    def __init__(
        self,
        alpha=0,  # Drift of the midprice
        sigma=0.5,  # Volatility of the midprice
        lambda_buy=2,  # Rate of buy market orders
        lambda_sell=2,  # Rate of sell market orders
        kappa_sell=27.0,  # Fill rate parameter for limit sell orders
        kappa_buy=27.0,  # Fill rate parameter for limit buy orders
        q_upper=3,  # Max inventory level
        q_lower=-3,  # Min inventory level
        Time=10.0,  # Terminal time in seconds
        dt=0.01,  # time step
        S0=10,  # Initial asset price
        X0=0,  # Initial cash balance
        Q0=0,  # Initial inventory
        phi_alpha=1,  # Uncertainty parameter of midprice drift
        phi_kappa=1,  # Uncertainty parameter of fill rate
        phi_lambda=1,  # Uncertainty parameter of rate of market orders
        phi=1e-4,  # Running penalty
        theta=1e-2,  # Terminal penalty
    ):
        self.alpha = alpha
        self.sigma = sigma
        self.lambda_buy = lambda_buy
        self.lambda_sell = lambda_sell
        self.kappa_buy = kappa_buy
        self.kappa_sell = kappa_sell
        self.q_max = q_upper
        self.q_min = q_lower
        self.T = Time
        self.dt = dt
        self.time_grid = np.linspace(0, self.T, int(self.T / self.dt) + 1)
        self.phi_alpha = phi_alpha
        self.phi_kappa = phi_kappa
        self.phi_lambda = phi_lambda
        self.phi = phi
        self.theta = theta
        self.length = len(self.time_grid)
        self.X0 = X0
        self.S0 = S0
        self.Q0 = Q0

        # initialise the state array
        self.stateX = np.zeros(self.length)
        self.stateS = np.zeros(self.length)
        self.stateQ = np.zeros(self.length)
        self.stateX[0] = X0
        self.stateS[0] = S0
        self.stateQ[0] = Q0
        self.statealpha = np.zeros(self.length)
        self.statekappabuy = np.zeros(self.length)
        self.statekappasell = np.zeros(self.length)
        self.statelambdabuy = np.zeros(self.length)
        self.statelambdasell = np.zeros(self.length)
        self.statealpha[0] = alpha
        self.statekappasell[0] = kappa_sell
        self.statekappabuy[0] = kappa_buy
        self.statelambdabuy[0] = lambda_buy
        self.statelambdasell[0] = lambda_sell

        self.objective = np.zeros(self.length)

        # denote current time step
        self.step = 0

    @property
    def ops_uncertainty(self):
        """
        Calcualte the optimal strategy with model uncertainty under the close-form expression
        """
        assert self.kappa_buy == self.kappa_sell, "kappa on both sides must be equal."
        assert (
            self.phi_lambda == self.phi_kappa
        ), "ambiguity of lambda and kappa must be equal."
        kappa = self.kappa_buy
        phi_uncertain = self.phi_lambda

        q_length = self.q_max - self.q_min + 1
        A = np.zeros([q_length, q_length])
        z = np.zeros(q_length)

        h_func = np.zeros([self.length, q_length])
        buy_depth = np.zeros([self.length, q_length - 1])
        sell_depth = np.zeros([self.length, q_length - 1])

        for i in range(q_length):
            # q = q_max - i
            z[i] = np.exp(-kappa * self.theta * (self.q_max - i) ** 2)

        for i in range(q_length):
            # i denote column
            for j in range(q_length):
                # j denote row
                if j == i:
                    # q = q_max - j
                    A[j, i] = (
                        self.alpha * kappa * (self.q_max - j)
                        - 0.5
                        * kappa
                        * self.phi_alpha
                        * self.sigma**2
                        * (self.q_max - j) ** 2
                    )
                elif j == i + 1:
                    # epsilon-
                    if phi_uncertain != 0:
                        A[j, i] = self.lambda_sell * (1 + phi_uncertain / kappa) ** (
                            -1 - kappa / phi_uncertain
                        )
                    else:
                        A[j, i] = self.lambda_sell / np.e
                elif j == i - 1:
                    # epsilon+
                    if phi_uncertain != 0:
                        A[j, i] = self.lambda_buy * (1 + phi_uncertain / kappa) ** (
                            -1 - kappa / phi_uncertain
                        )
                    else:
                        A[j, i] = self.lambda_buy / np.e

        for ti in range(self.length):
            h_func[ti, :] = np.log(np.dot(expm(A * (self.T - ti * self.dt)), z)) / kappa

        for i in range(q_length - 1):
            # q = q_max - j
            if phi_uncertain != 0:
                sell_depth[:, i] = (
                    np.log(1 + phi_uncertain / kappa) / phi_uncertain
                    + h_func[:, i]
                    - h_func[:, i + 1]
                )
            else:
                sell_depth[:, i] = 1 / kappa + h_func[:, i] - h_func[:, i + 1]

        for i in range(q_length - 1):
            if phi_uncertain != 0:
                buy_depth[:, i] = (
                    np.log(1 + phi_uncertain / kappa) / phi_uncertain
                    + h_func[:, i + 1]
                    - h_func[:, i]
                )
            else:
                buy_depth[:, i] = 1 / kappa + h_func[:, i + 1] - h_func[:, i]

        return sell_depth, buy_depth

    @property
    def ops_nouncertainty(self):
        """
        Calcualte the optimal strategy without model uncertainty under the close-form expression
        """
        assert self.kappa_buy == self.kappa_sell, "kappa on both sides must be equal."
        kappa = self.kappa_buy

        q_length = self.q_max - self.q_min + 1
        A = np.zeros([q_length, q_length])
        z = np.zeros(q_length)

        h_func = np.zeros([self.length, q_length])
        buy_depth = np.zeros([self.length, q_length - 1])
        sell_depth = np.zeros([self.length, q_length - 1])

        for i in range(q_length):
            # q = q_max - i
            z[i] = np.exp(-kappa * self.theta * (self.q_max - i) ** 2)

        for i in range(q_length):
            # i denote column
            for j in range(q_length):
                # j denote row
                if j == i:
                    # q = q_max - j
                    A[j, i] = (
                        self.alpha * kappa * (self.q_max - j)
                        - self.phi * kappa * (self.q_max - j) ** 2
                    )
                elif j == i + 1:
                    # epsilon-, lambda_sell(sell MOs, -)
                    A[j, i] = self.lambda_sell * np.e**-1
                elif j == i - 1:
                    # epsilon+, lambda_buy(buy MOs, +)
                    A[j, i] = self.lambda_buy * np.e**-1

        for ti in range(self.length):
            h_func[ti, :] = np.log(np.dot(expm(A * (self.T - ti * self.dt)), z)) / kappa

        for i in range(q_length - 1):
            # q = q_max - j
            sell_depth[:, i] = 1 / kappa + h_func[:, i] - h_func[:, i + 1]

        for i in range(q_length - 1):
            buy_depth[:, i] = 1 / kappa + h_func[:, i + 1] - h_func[:, i]

        sell_depth[sell_depth < 0] = 0
        buy_depth[buy_depth < 0] = 0

        return sell_depth, buy_depth

    @property
    def ops_uncertainty_num(self):

        # if self.kappa_buy == self.kappa_sell and self.phi_lambda == self.phi_kappa:
        #     return self.ops_uncertainty

        from scipy.integrate import solve_ivp
        from scipy.interpolate import interp1d

        q_length = self.q_max - self.q_min + 1
        # the limit exists even though denominator 0 (phi_kappa and phi_lambda 0)
        self.phi_kappa = 1e-12 if self.phi_kappa == 0 else self.phi_kappa
        self.phi_lambda = 1e-12 if self.phi_lambda == 0 else self.phi_lambda

        # B_p = (
        #     self.phi_kappa
        #     / (self.phi_kappa + self.kappa_sell)
        #     * np.exp(
        #         -self.kappa_sell
        #         / self.phi_kappa
        #         * np.log(1 + self.phi_kappa / self.kappa_sell)
        #     )
        # )
        B_p = (
            self.phi_kappa
            / (self.phi_kappa + self.kappa_sell)
            * (self.kappa_sell / (self.phi_kappa + self.kappa_sell))
            ** (self.kappa_sell / self.phi_kappa)
        )

        # B_n = (
        #     self.phi_kappa
        #     / (self.phi_kappa + self.kappa_buy)
        #     * np.exp(
        #         -self.kappa_buy
        #         / self.phi_kappa
        #         * np.log(1 + self.phi_kappa / self.kappa_buy)
        #     )
        # )
        B_n = (
            self.phi_kappa
            / (self.phi_kappa + self.kappa_buy)
            * (self.kappa_buy / (self.phi_kappa + self.kappa_buy))
            ** (self.kappa_buy / self.phi_kappa)
        )

        def fun(t, y):
            f = np.zeros_like(y)

            for i in range(1, len(y) - 1):
                # h is from q_max to q_min
                # q = q_max - i
                f[i] = 0.5 * self.phi_alpha * (
                    self.q_max - i
                ) ** 2 * self.sigma**2 - self.alpha * (self.q_max - i)

                if (
                    np.log(1 + self.phi_kappa / self.kappa_sell) / self.phi_kappa
                    > y[i + 1] - y[i]
                ):
                    f[i] -= (
                        self.lambda_buy
                        / self.phi_lambda
                        * (
                            1
                            - np.exp(
                                self.phi_lambda
                                / self.phi_kappa
                                * np.log(
                                    1
                                    - B_p * np.exp(self.kappa_sell * (y[i + 1] - y[i]))
                                )
                            )
                        )
                    )
                else:
                    f[i] -= (
                        self.lambda_buy
                        / self.phi_lambda
                        * (1 - np.exp(-self.phi_lambda * (y[i + 1] - y[i])))
                    )

                if (
                    np.log(1 + self.phi_kappa / self.kappa_buy) / self.phi_kappa
                    > y[i - 1] - y[i]
                ):
                    f[i] -= (
                        self.lambda_sell
                        / self.phi_lambda
                        * (
                            1
                            - np.exp(
                                self.phi_lambda
                                / self.phi_kappa
                                * np.log(
                                    1 - B_n * np.exp(self.kappa_buy * (y[i - 1] - y[i]))
                                )
                            )
                        )
                    )
                else:
                    f[i] -= (
                        self.lambda_sell
                        / self.phi_lambda
                        * (1 - np.exp(-self.phi_lambda * (y[i - 1] - y[i])))
                    )

            # f[q_max]
            if (
                np.log(1 + self.phi_kappa / self.kappa_sell) / self.phi_kappa
                > y[1] - y[0]
            ):
                f[0] = (
                    0.5 * self.phi_alpha * (self.q_max) ** 2 * self.sigma**2
                    - self.alpha * (self.q_max)
                    - self.lambda_buy
                    / self.phi_lambda
                    * (
                        1
                        - np.exp(
                            self.phi_lambda
                            / self.phi_kappa
                            * np.log(1 - B_p * np.exp(self.kappa_sell * (y[1] - y[0])))
                        )
                    )
                )
            else:
                f[0] = (
                    0.5 * self.phi_alpha * (self.q_max) ** 2 * self.sigma**2
                    - self.alpha * (self.q_max)
                    - self.lambda_buy
                    / self.phi_lambda
                    * (1 - np.exp(-self.phi_lambda * (y[1] - y[0])))
                )

            # f[q_min]
            if (
                np.log(1 + self.phi_kappa / self.kappa_buy) / self.phi_kappa
                > y[-2] - y[-1]
            ):
                f[-1] = (
                    0.5 * self.phi_alpha * (self.q_min) ** 2 * self.sigma**2
                    - self.alpha * (self.q_min)
                    - self.lambda_sell
                    / self.phi_lambda
                    * (
                        1
                        - np.exp(
                            self.phi_lambda
                            / self.phi_kappa
                            * np.log(1 - B_n * np.exp(self.kappa_buy * (y[-2] - y[-1])))
                        )
                    )
                )
            else:
                f[-1] = (
                    0.5 * self.phi_alpha * (self.q_min) ** 2 * self.sigma**2
                    - self.alpha * (self.q_min)
                    - self.lambda_sell
                    / self.phi_lambda
                    * (1 - np.exp(-self.phi_lambda * (y[-2] - y[-1])))
                )

            return f

        # Terminal condition
        h_T = np.zeros(q_length)
        for i in range(q_length):
            # q = q_max - i
            h_T[i] = -self.theta * (self.q_max - i) ** 2

        buy_depth = np.zeros([self.length, q_length - 1])
        sell_depth = np.zeros([self.length, q_length - 1])

        # # Solve the coupled system of ODEs from T to t=0
        # sol = solve_ivp(fun, [self.T, 0], h_T, t_eval=self.time_grid[::-1], method="RK45")

        # # sol.t, sol.y
        # for i in range(q_length - 1):
        #     sell_depth[:, i] = (
        #         np.log(1 + self.phi_kappa / self.kappa_sell) / self.phi_kappa
        #         + sol.y[i]
        #         - sol.y[i + 1]
        #     )

        # for i in range(q_length - 1):
        #     buy_depth[:, i] = (
        #         np.log(1 + self.phi_kappa / self.kappa_buy) / self.phi_kappa
        #         + sol.y[i + 1]
        #         - sol.y[i]
        #     )
        # return sell_depth[::-1], buy_depth[::-1]

        h = np.zeros([self.length, q_length])
        # Solve the coupled system of ODEs from T to t=0
        time_grid = np.linspace(0, self.T, int(1e5 + 1))

        sol = solve_ivp(
            fun,
            [self.T, 0],
            h_T,
            t_eval=time_grid[::-1],
            method="LSODA",
            rtol=1e-6,
            atol=1e-9,
        )
        # rtol at
        # Extract time_grid
        for i in range(q_length):
            # Create interpolation function for each h[q] = sol.y[q]
            interp_func = interp1d(sol.t, sol.y[i], kind="linear")
            # Interpolate at the final desired output times
            h[:, i] = interp_func(self.time_grid)

        for i in range(q_length - 1):
            sell_depth[:, i] = (
                np.log(1 + self.phi_kappa / self.kappa_sell) / self.phi_kappa
                + h[:, i]
                - h[:, i + 1]
            )

        sell_depth[sell_depth < 0] = 0

        for i in range(q_length - 1):
            buy_depth[:, i] = (
                np.log(1 + self.phi_kappa / self.kappa_buy) / self.phi_kappa
                + h[:, i + 1]
                - h[:, i]
            )

        buy_depth[buy_depth < 0] = 0

        return sell_depth, buy_depth

    def run(self, control_uncertainty=True, **kwargs):
        """
        Run the simulation by using the control either with or without model uncertainty
            control_unvcertainty: Bool. If Ture, use the optimal control with the model uncertainy, vice versa.

            **kwargs:
                pass the true environment parameters, such as
                    lambda_sell, lambda_buy, kappa_buy, kappa_sell, alpha
        """

        env_alpha = kwargs.get("env_alpha", self.alpha)
        env_lambda_sell = kwargs.get("env_lambda_sell", self.lambda_sell)
        env_lambda_buy = kwargs.get("env_lambda_buy", self.lambda_buy)
        env_kappa_sell = kwargs.get("env_kappa_sell", self.kappa_sell)
        env_kappa_buy = kwargs.get("env_kappa_buy", self.kappa_buy)

        if control_uncertainty:
            control = self.ops_uncertainty_num
        else:
            control = self.ops_nouncertainty

        # create nparray to log the number of coming MOs
        self.coming_sellMOs = np.zeros(self.length - 1)
        self.coming_buyMOs = np.zeros(self.length - 1)
        # create nparray to log the hit number of LOs posted by agent
        self.hitsellLOs = np.zeros(self.length - 1)
        self.hitbuyLOs = np.zeros(self.length - 1)
        # create nparray to log the depth posted by agent in each interval (batch_size, N)
        self.postselldepth = np.zeros(self.length - 1)
        self.postbuydepth = np.zeros(self.length - 1)

        for idx, t in enumerate(self.time_grid[1:]):
            # update time step
            self.step += 1

            Q_old = self.stateQ[idx]
            S_old = self.stateS[idx]
            X_old = self.stateX[idx]

            brownian_increments = np.random.randn() * np.sqrt(self.dt)

            # update midprice
            S_new = S_old + env_alpha * self.dt + self.sigma * brownian_increments

            # sell_depth = (
            #     control[0][idx, int(self.q_max + Q_old - 1)]
            #     if Q_old > self.q_min
            #     else 1e20
            # )
            # buy_depth = (
            #     control[1][idx, int(self.q_min + Q_old)] if Q_old < self.q_max else 1e20
            # )
            sell_depth = (
                control[0][idx + 1, int(self.q_max - Q_old)]
                if Q_old > self.q_min
                else 1e20
            )
            buy_depth = (
                control[1][idx + 1, int(self.q_max - Q_old - 1)]
                if Q_old < self.q_max
                else 1e20
            )

            self.postselldepth[idx] = sell_depth
            self.postbuydepth[idx] = buy_depth

            prob_sellside = np.exp(-sell_depth * env_kappa_sell)
            prob_buyside = np.exp(-buy_depth * env_kappa_buy)
            prob_sellside = 1 if prob_sellside > 1 else prob_sellside
            prob_buyside = 1 if prob_buyside > 1 else prob_buyside

            # update coming market orders as Poisson process
            sellMOs = np.random.poisson(env_lambda_sell * self.dt)
            buyMOs = np.random.poisson(env_lambda_buy * self.dt)
            self.coming_sellMOs[idx] = sellMOs
            self.coming_buyMOs[idx] = buyMOs

            # update the hit LOs posted by Agent
            dN_sell = np.random.binomial(buyMOs, prob_sellside)
            dN_buy = np.random.binomial(sellMOs, prob_buyside)
            self.hitsellLOs[idx] = dN_sell
            self.hitbuyLOs[idx] = dN_buy

            # update inventory
            Q_new = Q_old + dN_buy - dN_sell

            # update cash
            X_new = (
                X_old + (S_old + sell_depth) * dN_sell - (S_old - buy_depth) * dN_buy
            )

            # update state
            self.stateQ[idx + 1] = Q_new
            self.stateS[idx + 1] = S_new
            self.stateX[idx + 1] = X_new

            # calculate objective function
            objective_value = self.reward(current_step=idx + 1)
            self.objective[idx + 1] = objective_value

        return None

    def reward(self, current_step):
        """
        Calculate the objective function given current states
        """
        # current time step, step starts from 1
        X_T = self.stateX[current_step]
        S_T = self.stateS[current_step]
        Q_T = self.stateQ[current_step]

        return X_T + S_T * Q_T - self.theta * Q_T**2

    def complex_env_sim(
        self,
        epsilon_plus=0.001,
        epsilon_minus=0.001,
        beta_alpha=1,
        theta_lambda=0.2,
        beta_lambda=70 / 9,
        eta_lambda=5,
        nu_lambda=2,
        theta_kappa=15,
        beta_kappa=7 / 6,
        eta_kappa=5,
        nu_kappa=2,
        control_uncertainty=True,
    ):
        """
        Create a complex environment with the following dynamics:
            d S_t = alpha dt + sigma dW_t,
            d alpha_t = -beta_alpha alpha_t dt + epsilon^+ dM_t^+ - epsilon^- dM_t^-,
            d lambda_t^\pm = beta_lambda (theta_lambda - lambda_t^\pm )dt + eta_lambda dM_t^\pm + nu_lambda dM_t^\mp,
            d kappa_t^\pm = beta_kappa (theta_kappa - kappa_t^\pm )dt + eta_kappa dM_t^\pm + nu_kappa dM_t^\mp,
        """

        if control_uncertainty:
            control = self.ops_uncertainty_num
        else:
            control = self.ops_nouncertainty

        # create nparray to log the number of coming MOs
        self.coming_sellMOs = np.zeros(self.length - 1)
        self.coming_buyMOs = np.zeros(self.length - 1)
        # create nparray to log the hit number of LOs posted by agent
        self.hitsellLOs = np.zeros(self.length - 1)
        self.hitbuyLOs = np.zeros(self.length - 1)
        # create nparray to log the depth posted by agent in each interval (batch_size, N)
        self.postselldepth = np.zeros(self.length - 1)
        self.postbuydepth = np.zeros(self.length - 1)

        for idx, t in enumerate(self.time_grid[1:]):
            # update time step
            self.step += 1

            Q_old = self.stateQ[idx]
            S_old = self.stateS[idx]
            X_old = self.stateX[idx]

            alpha_old = self.statealpha[idx]
            kappa_sell_old = self.statekappasell[idx]
            kappa_buy_old = self.statekappabuy[idx]
            lambda_sell_old = self.statelambdasell[idx]
            lambda_buy_old = self.statelambdabuy[idx]

            sell_depth = (
                control[0][idx + 1, int(self.q_max - Q_old)]
                if Q_old > self.q_min
                else 1e20
            )
            buy_depth = (
                control[1][idx + 1, int(self.q_max - Q_old - 1)]
                if Q_old < self.q_max
                else 1e20
            )

            self.postselldepth[idx] = sell_depth
            self.postbuydepth[idx] = buy_depth

            prob_sellside = np.exp(-sell_depth * kappa_sell_old)
            prob_buyside = np.exp(-buy_depth * kappa_buy_old)
            prob_sellside = 1 if prob_sellside > 1 else prob_sellside
            prob_buyside = 1 if prob_buyside > 1 else prob_buyside

            # update coming market orders as Poisson process
            sellMOs = np.random.poisson(lambda_sell_old * self.dt)
            buyMOs = np.random.poisson(lambda_buy_old * self.dt)
            self.coming_sellMOs[idx] = sellMOs
            self.coming_buyMOs[idx] = buyMOs

            # update the hit LOs posted by Agent
            dN_sell = np.random.binomial(buyMOs, prob_sellside)
            dN_buy = np.random.binomial(sellMOs, prob_buyside)
            self.hitsellLOs[idx] = dN_sell
            self.hitbuyLOs[idx] = dN_buy

            # update dynamics
            brownian_increments = np.random.randn() * np.sqrt(self.dt)
            # update midprice
            S_new = S_old + alpha_old * self.dt + self.sigma * brownian_increments
            # update alpha
            alpha_new = (
                alpha_old
                - beta_alpha * alpha_old * self.dt
                + epsilon_plus * buyMOs
                - epsilon_minus * sellMOs
            )
            # update lambda_sell (lambda^-) and lambda_buy (lambda^+)
            lambda_sell_new = (
                lambda_sell_old
                + beta_lambda * (theta_lambda - lambda_sell_old) * self.dt
                + eta_lambda * sellMOs
                + nu_lambda * buyMOs
            )
            lambda_buy_new = (
                lambda_buy_old
                + beta_lambda * (theta_lambda - lambda_buy_old) * self.dt
                + eta_lambda * buyMOs
                + nu_lambda * sellMOs
            )
            # update kappa_sell (kappa^+) and kappa_buy (kappa^-)
            kappa_sell_new = (
                kappa_sell_old
                + beta_kappa * (theta_kappa - kappa_sell_old) * self.dt
                + eta_kappa * buyMOs
                + nu_kappa * sellMOs
            )
            kappa_buy_new = (
                kappa_buy_old
                + beta_kappa * (theta_kappa - kappa_buy_old) * self.dt
                + eta_kappa * sellMOs
                + nu_kappa * buyMOs
            )
            # update inventory
            Q_new = Q_old + dN_buy - dN_sell
            # update cash
            X_new = (
                X_old + (S_old + sell_depth) * dN_sell - (S_old - buy_depth) * dN_buy
            )

            # update state
            self.stateQ[idx + 1] = Q_new
            self.stateS[idx + 1] = S_new
            self.stateX[idx + 1] = X_new
            self.statealpha[idx + 1] = alpha_new
            self.statekappasell[idx + 1] = kappa_sell_new
            self.statekappabuy[idx + 1] = kappa_buy_new
            self.statelambdasell[idx + 1] = lambda_sell_new
            self.statelambdabuy[idx + 1] = lambda_buy_new

            # calculate objective function
            objective_value = self.reward(current_step=idx + 1)
            self.objective[idx + 1] = objective_value

        return None

    def reset(self):
        """
        reset the agent's state to the initial state
        """
        self.stateX = np.zeros(self.length)
        self.stateS = np.zeros(self.length)
        self.stateQ = np.zeros(self.length)
        self.stateX[0] = self.X0
        self.stateS[0] = self.S0
        self.stateQ[0] = self.Q0
        self.statealpha = np.zeros(self.length)
        self.statekappabuy = np.zeros(self.length)
        self.statekappasell = np.zeros(self.length)
        self.statelambdabuy = np.zeros(self.length)
        self.statelambdasell = np.zeros(self.length)
        self.statealpha[0] = self.alpha
        self.statekappasell[0] = self.kappa_sell
        self.statekappabuy[0] = self.kappa_buy
        self.statelambdabuy[0] = self.lambda_buy
        self.statelambdasell[0] = self.lambda_sell

        self.objective = np.zeros(self.length)
        self.step = 0

        return None


class RobustAgent_batch(RobustAgent):
    """
    Robust agent implemented with batch processing.
    """

    def __init__(
        self,
        alpha=0.1,  # Drift of the midprice
        sigma=0.01,  # Volatility of the midprice
        lambda_buy=2,  # Rate of buy market orders
        lambda_sell=2,  # Rate of sell market orders
        kappa_buy=27.0,  # Fill rate parameter for limit buy orders
        kappa_sell=27.0,  # Fill rate parameter for limit sell orders
        q_upper=3,  # Max inventory level
        q_lower=-3,  # Min inventory level
        Time=1,  # Terminal time
        dt=0.01,  # Time step
        S0=10,  # Initial asset price
        X0=0,  # Initial cash balance
        Q0=0,  # Initial inventory
        phi_alpha=1,  # Uncertainty parameter of midprice drift
        phi_kappa=1,  # Uncertainty parameter of fill rate
        phi_lambda=1,  # Uncertainty parameter of rate of market orders
        phi=1e-4,  # Running penalty
        theta=1e-2,  # Terminal penalty
        MC_samples=10000,  # Number of MC_samples in a batch
    ):
        super().__init__(
            alpha=alpha,
            sigma=sigma,
            lambda_buy=lambda_buy,
            lambda_sell=lambda_sell,
            kappa_buy=kappa_buy,
            kappa_sell=kappa_sell,
            q_upper=q_upper,
            q_lower=q_lower,
            Time=Time,
            dt=dt,
            S0=S0,
            X0=X0,
            Q0=Q0,
            phi_alpha=phi_alpha,
            phi_kappa=phi_kappa,
            phi_lambda=phi_lambda,
            phi=phi,
            theta=theta,
        )
        self.MC_samples = MC_samples
        self.reset_initial()

    def reset_initial(self):
        """
        reset the agent's intial state as a batch
        """
        self.stateX = np.zeros([self.MC_samples, self.length])
        self.stateS = np.zeros([self.MC_samples, self.length])
        self.stateQ = np.zeros([self.MC_samples, self.length])
        self.stateX[:, 0] = self.X0
        self.stateS[:, 0] = self.S0
        self.stateQ[:, 0] = self.Q0

        self.statealpha = np.zeros([self.MC_samples, self.length])
        self.statekappabuy = np.zeros([self.MC_samples, self.length])
        self.statekappasell = np.zeros([self.MC_samples, self.length])
        self.statelambdabuy = np.zeros([self.MC_samples, self.length])
        self.statelambdasell = np.zeros([self.MC_samples, self.length])
        self.statealpha[:, 0] = self.alpha
        self.statekappasell[:, 0] = self.kappa_sell
        self.statekappabuy[:, 0] = self.kappa_buy
        self.statelambdabuy[:, 0] = self.lambda_buy
        self.statelambdasell[:, 0] = self.lambda_sell

        self.objective = np.zeros([self.MC_samples, self.length])

        return None

    def complex_env_sim_batch(
        self,
        control_uncertainty=True,
        epsilon_plus=0.001,
        epsilon_minus=0.001,
        beta_alpha=1,
        theta_lambda=0.2,
        beta_lambda=70 / 9,
        eta_lambda=5,
        nu_lambda=2,
        theta_kappa=15,
        beta_kappa=7 / 6,
        eta_kappa=5,
        nu_kappa=2,
    ):
        """
        Create a complex environment with the following dynamics:
            d S_t = alpha dt + sigma dW_t,
            d alpha_t = -beta_alpha alpha_t dt + epsilon^+ dM_t^+ - epsilon^- dM_t^-,
            d lambda_t^\pm = beta_lambda (theta_lambda - lambda_t^\pm )dt + eta_lambda dM_t^\pm + nu_lambda dM_t^\mp,
            d kappa_t^\pm = beta_kappa (theta_kappa - kappa_t^\pm )dt + eta_kappa dM_t^\pm + nu_kappa dM_t^\mp,
        """

        if control_uncertainty:
            control = self.ops_uncertainty_num
        else:
            control = self.ops_nouncertainty

        # if q is q_max and q_low, the depth is 1e20
        optimal_sell = np.insert(control[0], -1, np.array([1e20]*self.length), axis=1)
        optimal_buy = np.insert(control[1], 0, np.array([1e20]*self.length), axis=1)

        # create nparray to log the number of coming MOs
        self.coming_sellMOs = np.zeros([self.MC_samples, self.length-1])
        self.coming_buyMOs = np.zeros([self.MC_samples, self.length-1])
        # create nparray to log the hit number of LOs posted by agent
        self.hitsellLOs = np.zeros([self.MC_samples, self.length-1])
        self.hitbuyLOs = np.zeros([self.MC_samples, self.length-1])
        # create nparray to log the depth posted by agent in each interval (batch_size, N)
        self.postselldepth = np.zeros([self.MC_samples, self.length-1])
        self.postbuydepth = np.zeros([self.MC_samples, self.length-1])

        for idx, t in enumerate(self.time_grid[:-1]):
            # update time step
            self.step += 1

            # Current time step's control matrix (batch_size, q_upper - q_lower + 1 )
            sell_depth_matrix = optimal_sell[np.newaxis, idx, :].repeat(
                self.MC_samples, axis=0
            )
            buy_depth_matrix = optimal_buy[np.newaxis, idx, :].repeat(
                self.MC_samples, axis=0
            ) 

            Q_old = self.stateQ[:, idx]
            S_old = self.stateS[:, idx]
            X_old = self.stateX[:, idx]

            alpha_old = self.statealpha[:, idx]
            kappa_sell_old = self.statekappasell[:, idx]
            kappa_buy_old = self.statekappabuy[:, idx]
            lambda_sell_old = self.statelambdasell[:, idx]
            lambda_buy_old = self.statelambdabuy[:, idx]

            # indices of control, (batch_size, 1) 
            # if Q_old is q_max then the buy depth is 1e20
            # if Q_old is q_min then the sell depth is 1e20
            indices_sell = (self.q_max - Q_old).astype(int)
            indices_buy = (self.q_max - Q_old).astype(int)

            # if t == self.time_grid[-2]:
            #     breakpoint()
            # update depth
            sell_depth = np.take_along_axis(sell_depth_matrix, indices_sell.reshape(self.MC_samples, -1), axis=1).flatten()
            buy_depth = np.take_along_axis(buy_depth_matrix, indices_buy.reshape(self.MC_samples, -1), axis=1).flatten()
            # constraints on depth          
            sell_depth = sell_depth + (Q_old <= self.q_min) * 1e20
            buy_depth = buy_depth + (Q_old >= self.q_max) * 1e20
            # log the depth
            self.postselldepth[:, idx] = sell_depth
            self.postbuydepth[:, idx] = buy_depth

            # update coming market orders as Poisson process
            buy_MOs = np.random.poisson(lambda_buy_old * self.dt, size=self.MC_samples)
            sell_MOs = np.random.poisson(lambda_sell_old * self.dt, size=self.MC_samples)
            self.coming_sellMOs[:, idx] = sell_MOs
            self.coming_buyMOs[:, idx] = buy_MOs
            # update the hit LOs posted by Agent
            prob_sellside = np.exp(-sell_depth * kappa_sell_old)
            prob_buyside = np.exp(-buy_depth * kappa_buy_old)

            dN_sell = np.random.binomial(buy_MOs, prob_sellside)
            dN_buy = np.random.binomial(sell_MOs, prob_buyside)
            self.hitsellLOs[:, idx] = dN_sell
            self.hitbuyLOs[:, idx] = dN_buy
            
            # update dynamics
            brownian_increments = np.random.randn(self.MC_samples) * np.sqrt(self.dt)
            # update midprice
            S_new = S_old + alpha_old * self.dt + self.sigma * brownian_increments

            # update alpha
            alpha_new = (
                alpha_old
                - beta_alpha * alpha_old * self.dt
                + epsilon_plus * buy_MOs
                - epsilon_minus * sell_MOs
            )
            # update lambda_sell (lambda^-) and lambda_buy (lambda^+)
            lambda_sell_new = (
                lambda_sell_old
                + beta_lambda * (theta_lambda - lambda_sell_old) * self.dt
                + eta_lambda * sell_MOs
                + nu_lambda * buy_MOs
            )
            lambda_buy_new = (
                lambda_buy_old
                + beta_lambda * (theta_lambda - lambda_buy_old) * self.dt
                + eta_lambda * buy_MOs
                + nu_lambda * sell_MOs
            )
            # update kappa_sell (kappa^+) and kappa_buy (kappa^-)
            kappa_sell_new = (
                kappa_sell_old
                + beta_kappa * (theta_kappa - kappa_sell_old) * self.dt
                + eta_kappa * buy_MOs
                + nu_kappa * sell_MOs
            )
            kappa_buy_new = (
                kappa_buy_old
                + beta_kappa * (theta_kappa - kappa_buy_old) * self.dt
                + eta_kappa * sell_MOs
                + nu_kappa * buy_MOs
            )
            # update inventory
            Q_new = Q_old + dN_buy - dN_sell
            # constaints on inventory
            Q_new = np.clip(Q_new, self.q_min, self.q_max)

            # update cash
            X_new = (
                X_old + (S_old + sell_depth) * dN_sell - (S_old - buy_depth) * dN_buy
            )

            # update state
            self.stateQ[:, idx + 1] = Q_new
            self.stateS[:, idx + 1] = S_new
            self.stateX[:, idx + 1] = X_new
            self.statealpha[:, idx + 1] = alpha_new
            self.statekappasell[:, idx + 1] = kappa_sell_new
            self.statekappabuy[:, idx + 1] = kappa_buy_new
            self.statelambdasell[:, idx + 1] = lambda_sell_new
            self.statelambdabuy[:, idx + 1] = lambda_buy_new

            # calculate objective function
            objective_value = self.reward_batch(current_step=idx + 1)
            self.objective[:, idx + 1] = objective_value

        return None
    
    def reward_batch(self, current_step):
        """
        Calculate the objective function given current states
        """
        # current time step, step starts from 1
        X_T = self.stateX[:, current_step]
        S_T = self.stateS[:, current_step]
        Q_T = self.stateQ[:, current_step]

        return X_T + S_T * Q_T - self.theta * Q_T**2
    

    def complex_env_sim_batch_lowmemory(
        self,
        control_uncertainty=True,
        epsilon_plus=0.001,
        epsilon_minus=0.001,
        beta_alpha=1,
        theta_lambda=0.2,
        beta_lambda=70 / 9,
        eta_lambda=5,
        nu_lambda=2,
        theta_kappa=15,
        beta_kappa=7 / 6,
        eta_kappa=5,
        nu_kappa=2,
    ):
        """
        Create a complex environment with the following dynamics:
            d S_t = alpha dt + sigma dW_t,
            d alpha_t = -beta_alpha alpha_t dt + epsilon^+ dM_t^+ - epsilon^- dM_t^-,
            d lambda_t^\pm = beta_lambda (theta_lambda - lambda_t^\pm )dt + eta_lambda dM_t^\pm + nu_lambda dM_t^\mp,
            d kappa_t^\pm = beta_kappa (theta_kappa - kappa_t^\pm )dt + eta_kappa dM_t^\pm + nu_kappa dM_t^\mp,
        """

        if control_uncertainty:
            control = self.ops_uncertainty_num
        else:
            control = self.ops_nouncertainty

        # if q is q_max and q_low, the depth is 1e20
        optimal_sell = np.insert(control[0], -1, np.array([1e20]*self.length), axis=1)
        optimal_buy = np.insert(control[1], 0, np.array([1e20]*self.length), axis=1)

        Q_old = self.stateQ[:, 0]
        S_old = self.stateS[:, 0]
        X_old = self.stateX[:, 0]

        alpha_old = self.statealpha[:, 0]
        kappa_sell_old = self.statekappasell[:, 0]
        kappa_buy_old = self.statekappabuy[:, 0]
        lambda_sell_old = self.statelambdasell[:, 0]
        lambda_buy_old = self.statelambdabuy[:, 0]

        for idx, t in enumerate(self.time_grid[:-1]):
            # update time step

            # Current time step's control matrix (batch_size, q_upper - q_lower + 1 )
            sell_depth_matrix = optimal_sell[np.newaxis, idx, :].repeat(
                self.MC_samples, axis=0
            )
            buy_depth_matrix = optimal_buy[np.newaxis, idx, :].repeat(
                self.MC_samples, axis=0
            ) 

            # indices of control, (batch_size, 1) 
            # if Q_old is q_max then the buy depth is 1e20
            # if Q_old is q_min then the sell depth is 1e20
            indices_sell = (self.q_max - Q_old).astype(int)
            indices_buy = (self.q_max - Q_old).astype(int)

            # update depth
            sell_depth = np.take_along_axis(sell_depth_matrix, indices_sell.reshape(self.MC_samples, -1), axis=1).flatten()
            buy_depth = np.take_along_axis(buy_depth_matrix, indices_buy.reshape(self.MC_samples, -1), axis=1).flatten()
            # constraints on depth          
            sell_depth = sell_depth + (Q_old <= self.q_min) * 1e20
            buy_depth = buy_depth + (Q_old >= self.q_max) * 1e20

            # update coming market orders as Poisson process
            buy_MOs = np.random.poisson(lambda_buy_old * self.dt, size=self.MC_samples)
            sell_MOs = np.random.poisson(lambda_sell_old * self.dt, size=self.MC_samples)

            # update the hit LOs posted by Agent
            prob_sellside = np.exp(-sell_depth * kappa_sell_old)
            prob_buyside = np.exp(-buy_depth * kappa_buy_old)

            dN_sell = np.random.binomial(buy_MOs, prob_sellside)
            dN_buy = np.random.binomial(sell_MOs, prob_buyside)
            
            # update dynamics
            brownian_increments = np.random.randn(self.MC_samples) * np.sqrt(self.dt)
            # update midprice
            S_new = S_old + alpha_old * self.dt + self.sigma * brownian_increments

            # update alpha
            alpha_new = (
                alpha_old
                - beta_alpha * alpha_old * self.dt
                + epsilon_plus * buy_MOs
                - epsilon_minus * sell_MOs
            )
            # update lambda_sell (lambda^-) and lambda_buy (lambda^+)
            lambda_sell_new = (
                lambda_sell_old
                + beta_lambda * (theta_lambda - lambda_sell_old) * self.dt
                + eta_lambda * sell_MOs
                + nu_lambda * buy_MOs
            )
            lambda_buy_new = (
                lambda_buy_old
                + beta_lambda * (theta_lambda - lambda_buy_old) * self.dt
                + eta_lambda * buy_MOs
                + nu_lambda * sell_MOs
            )
            # update kappa_sell (kappa^+) and kappa_buy (kappa^-)
            kappa_sell_new = (
                kappa_sell_old
                + beta_kappa * (theta_kappa - kappa_sell_old) * self.dt
                + eta_kappa * buy_MOs
                + nu_kappa * sell_MOs
            )
            kappa_buy_new = (
                kappa_buy_old
                + beta_kappa * (theta_kappa - kappa_buy_old) * self.dt
                + eta_kappa * sell_MOs
                + nu_kappa * buy_MOs
            )
            # update inventory
            Q_new = Q_old + dN_buy - dN_sell
            # constaints on inventory
            Q_new = np.clip(Q_new, self.q_min, self.q_max)

            # update cash
            X_new = (
                X_old + (S_old + sell_depth) * dN_sell - (S_old - buy_depth) * dN_buy
            )

            # update state
            Q_old = Q_new
            S_old = S_new
            X_old = X_new
            alpha_old = alpha_new
            kappa_sell_old = kappa_sell_new
            kappa_buy_old = kappa_buy_new
            lambda_sell_old = lambda_sell_new
            lambda_buy_old = lambda_buy_new

        # calculate the terminal objective function
        objective_value = (X_new + S_new * Q_new - self.theta * Q_new**2) - (self.stateX[:, 0] + self.stateS[:, 0] * self.stateQ[:, 0] - self.theta * self.stateQ[:, 0]**2)

        return objective_value




