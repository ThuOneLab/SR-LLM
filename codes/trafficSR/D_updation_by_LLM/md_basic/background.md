The car-following model is used to describe the acceleration of the following car in driving scenarios when it is
following the leading car. I will first provide you with two classic car-following models, and explain the physical
meaning of these two car-following models.

The known IDM car-following model is expressed as:
$$
a_{IDM}=\alpha\times[1-(\frac{v}{v_0})^4-(\frac{s_0}{s})^2]
$$
where $\alpha$ is the maximum acceleration of the following car, v is the speed of the following car, $v_0$ is the
desired speed of the following car, $s_0$ is the desired following distance, and $s$ is the distance between the
following car and the leading car (i.e., the following distance). In the IDM model, $\frac{v}{v_0}$ describes the gap
between the actual speed of the following car and the desired speed. According to the above formula, when the driving
speed $v$ of the following car is much smaller than the desired speed, the final acceleration value will be larger,
reflecting that the driver will increase the acceleration when driving slowly; when the driving speed $v$ of the
following car is much larger than the desired speed, the acceleration value will be smaller, reflecting that the driver
will decrease the acceleration when driving fast. $\frac{s_0}{s}$ describes the gap between the desired following
distance and the actual following distance. The larger the actual following distance $s$, the smaller $\frac{s_0}{s}$,
indicating that the driver will take a larger acceleration; when the actual following distance $s$ is very small,
$\frac{s_0}{s}$ will be very large, reflecting that the acceleration of the following car will become very small, even
negative, and the following car enters a deceleration state.

The known GHR car-following model is expressed as:
$$
a_{GHR}=k\times\frac{v\times\Delta{v}}{s}
$$
where $k$ is a constant parameter, v is the speed of the following car, $\Delta{v}$ is the speed difference between the
leading car and the following car, and s is the distance between the following car and the leading car. In the GHR
model, k reflects the sensitivity of acceleration changes when different drivers are driving; the fraction item in the
back reflects the impact of the following car speed, the speed difference between the leading and following cars, and
the following distance on the acceleration. The larger the speed of the following car, the larger the speed difference
between the leading and following cars, and the smaller the following distance, the more obvious the acceleration and
deceleration of the following car.