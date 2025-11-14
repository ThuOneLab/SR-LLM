import os
import sys
sys.path.append(os.path.abspath(os.getcwd()))
from codes.trafficSR.D_updation_by_LLM.RAG_Agent import RAG_AGENT
from codes.trafficSR.D_updation_by_LLM.Modules.Knowledge_Pool import Knowledge

port = "7890"
agent = RAG_AGENT(port=port, memory_path="../../../ragLibrary/memory_ngsim_find_new_reflection")
knowledge_length = len(agent.knowledge_pool.content._collection.get(include=['embeddings'])['embeddings'])
for _ in range(knowledge_length):
    agent.knowledge_pool.delete_knowledge([0])
'''--------------------------------------IDM'''
'''factor_v_ratio'''
agent.knowledge_pool.add_knowledge(
    Knowledge(
        source="IDM",
        key=r'$v$: The symbol in speed unit which represents ego vehicle speed, ' + \
            '$v_0$: The symbol in speed unit which represents desired ego vehicle speed',
        target=r'$factor_v_ratio$: The symbol in dimensionless units which represents the ratio of current speed to desired speed',
        content=
        r"In order to reflect the proportional relationship between the speed of the ego vehicle $v$ " + \
        "and the desired speed of ego vehicle $v_0$, " + \
        r"human experts use $v$ as the dividend and $v_0$ as the divisor to a new symbol #v/v_0#. " + \
        r"When $v$ is greater than $v_0$, this term will increase. " + \
        r"When $v$ is less than $v_0$, this term will decrease. " + \
        r"It can be further condensed as the influencing factor of speed, marked as $factor_v_ratio$.",
        comment=r"Good symbol that reflects the proportional relationship between the current speed and the desired " + \
                r"speed of ego vehicle. " + \
                "This operation also has excellent fitting performance.",
        reflection=r"It's a good symbol. I need to consider the current speed and the desired speed of ego vehicle."
    )
)
'''factor_v_squared'''
agent.knowledge_pool.add_knowledge(
    Knowledge(
        source="IDM",
        key=r'$factor_v_ratio$: The symbol in dimensionless units which represents the ratio of current speed to desired speed',
        target=r'$factor_v_squared$: The symbol in dimensionless unit which represents the square ratio of the speed influence factor',
        content=
        r"In the vehicle following formula, human experts often square the dimensionless unit variable to achieve a more accurate fitting effect. " + \
        r"Human experts square $factor_v_ratio$ to obtain a new symbol $factor_v_squared$, which is #n2(factor_v_ratio)#. " + \
        r"It can be further condensed as the influencing factor of speed, marked as $factor_v_squared$.",
        comment=r"When there are dimensionless symbols, operations such as square or root can be performed to obtain " + \
                r"more accurate symbols that are also dimensionless.",
        reflection=r"I will remember when encountering dimensionless symbols, trying to square or root them."
    )
)
'''factor_v_quadratic'''
agent.knowledge_pool.add_knowledge(
    Knowledge(
        source="IDM",
        key=r'$factor_v_squared$: The symbol in dimensionless unit which represents the square ratio of the speed influence factor',
        target=r'$factor_v_quadratic$: The symbol in dimensionless unit which represents the quadratic ratio of the speed influence factor',
        content=
        r"In the vehicle following formula, human experts often square the dimensionless unit variable to achieve a more accurate fitting effect. " + \
        r"Human experts square $factor_v_squared$ to obtain a new symbol $factor_v_quadratic$, which is #n2(factor_v_squared)#. " + \
        r"It can be further condensed as the influencing factor of speed, marked as $factor_v_quadratic$.",
        comment=r"When there are dimensionless symbols, operations such as square or root can be performed to obtain " + \
                r"more accurate symbols that are also dimensionless.",
        reflection=r"I will remember when encountering dimensionless symbols, trying to square or root them."
    )
)
'''s_equilibrium'''
agent.knowledge_pool.add_knowledge(
    Knowledge(
        source="IDM",
        key=r'$v$: The symbol in speed unit which represents ego vehicle speed, ' + \
            '$T$: The symbol in time unit which represents the desired time headway',
        target=r'$s_maintained$: The symbol in distance unit which represents the reserved spacing considering the ' + \
               'current speed and desired time headway',
        content=
        r"$T$ starts from the driver's perspective and indicates how long it takes for the current vehicle to reach " + \
        "its current position while maintaining its current speed if the preceding vehicle suddenly stops. " + \
        r"Human experts multiply $v$ with $T$ to obtain the reserved spacing #vT#." + \
        r"The higher the speed of ego vehicle, the greater the distance you need to maintain from the car in front of you. " + \
        r"This distance ensures that even if the speed of the preceding vehicle changes, " + \
        "the current vehicle has enough time and distance to react and adjust its speed to prevent rear end collisions. " + \
        r"This symbol can be named as $s_maintained$.",
        comment=r"Good symbol that reflects the distance that needs to be maintained between vehicles at the current speed. " + \
                "This operation also has excellent fitting performance.",
        reflection=r"I need to consider the relationship between ego vehicle speed and the required time headway."
    )
)
agent.knowledge_pool.add_knowledge(
    Knowledge(
        source="IDM",
        key=r'$s_maintained$: The symbol in distance unit which represents the reserved spa cing considering the ' + \
            'current speed and desired time headway, ' + \
            '$s_0$: The symbol in distance unit which represents the desired minimum following distance in stationary state',
        target=r'$s_equilibrium$: The symbol in distance unit which represents the desired spacing ' + \
               'in equilibrium car-following situations',
        content=
        r"Human experts add $s_0$ with $s_maintained$ to obtain the equilibrium desired spacing $s_equilibrium$." + \
        r"When following a leading vehicle, the distance to the leading vehicle is approximately equilibrium distance " + \
        r", written as #s_0 + s_maintained#. " + \
        r"When the speed of ego vehicle is 0, at least maintain minimum distance $s_0$. " + \
        r"If the speed of ego vehicle is greater than 0, the influence of maintaining the headway of the vehicle " + \
        r"should also be considered, which is $s_maintained$. " + \
        "Therefore, after adding up the two parts, it forms the distance that should be maintained in equilibrium " + \
        r"car-following situations. " + \
        r"This symbol can be named as $s_equilibrium$.",
        comment=r"Good symbol that reflects the distance that needs to be maintained of car-following scenario in equilibrium state.",
        reflection=r"I not only need to consider the minimum following distance $s_0$, but also need to meet $s_maintained$."
    )
)
'''s_dynamic'''
agent.knowledge_pool.add_knowledge(
    Knowledge(
        source="IDM",
        key=r'$v$: The symbol in speed unit which represents ego vehicle speed, ' + \
            r'$delta_v$: The symbol in speed unit which represents subtracting the speed of the preceding vehicle from the speed of ego vehicle',
        target=r'$v2_dynamic_adjustment$: The symbol in productSpeed unit which represents relative change in speed ' + \
               r'between the ego vehicle and the preceding vehicle',
        content=
        r"Human experts product of $v$ and $delta_v$ to obtain the comprehensive response capability item #v*delta_v#. " + \
        r"$delta_v$ is the result of subtracting the speed of the preceding vehicle from the speed of ego vehicle. "
        r"When $v$ or $delta_v$ increases, it means that the ego vehicle needs to adjust the distance further to cope with changes in speed and ensure a safe distance.. " + \
        r"It can be further condensed as the demand for safe vehicle distance due to vehicle speed and speed changes, marked as $a_response_capacity$.",
        comment=r"Good symbol that reflects the demand for dynamic adjustment of safety distance based on self driving speed and relative speed.",
        reflection=r"I need to consider ego vehicle speed and relative speed."
    )
)
agent.knowledge_pool.add_knowledge(
    Knowledge(
        source="IDM",
        key=r'$alpha$: The symbol in acceleration unit which represents the comfortable maximum acceleration of ego vehicle, ' + \
            r'$b$: The symbol in acceleration unit which represents comfortable maximum deceleration for the driver',
        target=r'$a_response_capacity$: The symbol in acceleration unit which represents the comprehensive response ' + \
               r'capability of ego vehicle',
        content=
        r"Human experts square root the product of $alpha$ and $b$ to obtain the comprehensive response capability item #sqrt(alpha*b)#. " + \
        r"This new symbol combines the vehicle's acceleration and deceleration capabilities, written as $a_response_capacity$."
        r"Larger value of $alpha$ and $b$ means that the vehicle has stronger acceleration and deceleration capabilities, and can respond in a shorter time and distance. " + \
        r"Its square root form ensures that this term is a positive value, and its impact on safety distance is non-linear, " + \
        r"reflecting the vehicle's responsiveness in the face of emergency situations. " + \
        r"It can be further condensed as the influencing factor of comprehensive response capability of ego vehicle on acceleration, marked as $a_response_capacity$.",
        comment=r"Good symbol that reflects the comprehensive response capability of ego vehicle.",
        reflection=r"I need to consider the comprehensive response capability of ego vehicle."
    )
)
agent.knowledge_pool.add_knowledge(
    Knowledge(
        source="IDM",
        key=r'$a_response_capacity$: The symbol in acceleration unit which represents the comprehensive response ' + \
            r'capability of ego vehicle',
        target=r'$a_dynamic_adjustment$: The symbol in acceleration unit which represents the dynamic adjustment ' + \
               r'capability of ego vehicle',
        content=
        r"Human experts add $a_response_capacity$ and $a_response_capacity$ to obtain the dynamic adjustment capability item #a_response_capacity+a_response_capacity#. " + \
        r"The reason for adding is because empirical adjustment coefficient 2, which is an adjustment factor obtained based on empirical formulas and actual testing. " + \
        r"After adding, #a_response_capacity+a_response_capacity# can better reflect the dynamic adjustment ability than $a_response_capacity$. " + \
        r"It can be further condensed as the influencing factor of dynamic adjustment capability of ego vehicle on acceleration, marked as $a_dynamic_adjustment$.",
        comment=r"Good symbol that reflects the dynamic adjustment capability of ego vehicle.",
        reflection=r"I need to consider the dynamic adjustment capability of ego vehicle. " + \
                   "And sometimes empirical adjustment coefficient needs to be adopted."
    )
)
agent.knowledge_pool.add_knowledge(
    Knowledge(
        source="IDM",
        key=r'$v2_dynamic_adjustment$: The symbol in productSpeed unit which represents relative change in speed ' + \
            'between the ego vehicle and the preceding vehicle, ' + \
            r'$a_dynamic_adjustment$: The symbol in acceleration unit which represents the dynamic adjustment ' + \
            'capability of ego vehicle',
        target=r'$s_dynamic$: The symbol in distance unit which represents the additional desired spacing ' + \
               'in emergency car-following situations, especially when there is a speed difference between the two vehicles',
        content=
        r"Human experts divide $v2_dynamic_adjustment$ with $a_dynamic_adjustment$ to obtain the dynamic desired spacing $s_dynamic$. " + \
        r"This is the additional safety distance required for the vehicle, which is calculated by taking into account " + \
        r"the current vehicle speed, speed difference with preceding vehicle, and the dynamic response capability of ego vehicle, " + \
        r"written as #v2_dynamic_adjustment/a_dynamic_adjustment#. " + \
        r"The introduction of additional distance is mainly to have enough space to slow down or stop in emergency situations (such as sudden braking of the preceding vehicle). " + \
        r"It can be further condensed as the additional desired spacing in emergency car-following situations, marked as $s_dynamic$.",
        comment=r"Good symbol that reflects the distance that needs to be maintained of car-following scenario in emergency response state.",
        reflection=r"I need to consider the additional distance required when there is a speed difference."
    )
)
'''s_safe'''
agent.knowledge_pool.add_knowledge(
    Knowledge(
        source="IDM",
        key=r'$s_equilibrium$: The symbol in distance unit which represents the desired spacing ' + \
            'in equilibrium car-following situations, ' + \
            r'$s_dynamic$: The symbol in distance unit which represents the additional desired spacing ' + \
            'in emergency car-following situations, especially when there is a speed difference between the two vehicles',
        target=r'$s_safe$: The symbol in distance unit which represents the desired safe distance that vehicles ' + \
               'should maintain in different situations',
        content=
        r"In order to take into account desired vehicle spacing under both equilibrium and emergency conditions, " + \
        r"human experts add these two parts to obtain a new symbol #s_equilibrium+s_dynamic#. " + \
        r"Thus, it can perform well at low speeds or at rest, and can dynamically adjust with speed, speed difference, " + \
        "dynamic response ability, etc. " + \
        r"It can be further condensed as the safe distance, marked as $s_save$.",
        comment=r"Good symbol that reflects different aspects required for safe vehicle distance.",
        reflection=r"I need to consider desired vehicle spacing under both equilibrium and emergency conditions."
    )
)
'''factor_s_ratio'''
agent.knowledge_pool.add_knowledge(
    Knowledge(
        source="IDM",
        key=r'$s_safe$: The symbol in distance unit which represents the desired safe distance that vehicles ' + \
            'should maintain in different situations, ' + \
            '$s$: The symbol in distance unit which represents the current following distance',
        target=r'$factor_s_ratio$: The symbol in dimensionless unit which represents the ratio of current following distance to desired safe distance',
        content=
        r"In order to reflect the proportional relationship between the desired safe distance of ego vehicle $s_safe$ " + \
        r"and the speed of the ego vehicle $s$, " + \
        r"human experts use $s_safe$ as the dividend and $s$ as the divisor to obtain a new symbol #s_safe/s#. " + \
        r"Please do not divide $s$ by $s_safe$."
        r"When $s_safe$ is greater than $s$, this term will increase. " + \
        r"When $s_safe$ is less than $s$, this term will decrease. " + \
        r"It can be further condensed as the influencing factor of distance, marked as $factor_s_ratio$.",
        comment=r"Good symbol that reflects the proportional relationship between the current distance and " + \
                r"the desired safe distance of ego vehicle.",
        reflection=r"I need to consider the proportional relationship between the current distance and the " + \
                   r"desired safe distance of ego vehicle."
    )
)

'''factor_s_squared'''
agent.knowledge_pool.add_knowledge(
    Knowledge(
        source="IDM",
        key=r'$factor_s_ratio$: The symbol in dimensionless unit which represents the ratio of current following distance to desired safe distance',
        target=r'$factor_s_squared$: The symbol in dimensionless unit which represents the square of the distance influence factor',
        content=
        r"In the vehicle following formula, human experts often square the dimensionless unit variable " + \
        r"to achieve a more accurate fitting effect. " + \
        r"Human experts square $factor_s_ratio$ to obtain a new symbol #n2(factor_s_ratio)#. " + \
        r"It can be further condensed as the influencing factor of distance, marked as $factor_s_squared$.",
        comment=r"When there are dimensionless symbols, operations such as square or root can be performed to obtain " + \
                "more accurate symbols that are also dimensionless.",
        reflection=r"I will remember when encountering dimensionless symbols, trying to square or root them."
    )
)
'''factor_integrated'''
agent.knowledge_pool.add_knowledge(
    Knowledge(
        source="IDM",
        key=r'$factor_v_quadratic$: The symbol in dimensionless unit which represents the quadratic of the speed ratio influence factor, ' + \
            r'$factor_s_squared$: The symbol in dimensionless unit which represents the square of the distance influence factor',
        target=r'$factor_integrated$: The symbol in dimensionless unit which represents both speed and distance influence factors',
        content=
        r"In the vehicle following formula, human experts often add dimensionless factor with other dimensionless " + \
        r"factor to comprehensively reflect the influence of multiple factors. " + \
        r"Human experts add $factor_v_quadratic$ and $factor_s_squared$ to obtain the impact of both aspects, which is #factor_v_quadratic+factor_s_squared#. " + \
        r"It can be further condensed as the influencing factor of both speed and distance influence, marked as $factor_integrated$.",
        comment=r"You can add dimensionless factors together to comprehensively demonstrate their impact " + \
                "on acceleration.",
        reflection=r"I will remember when encountering dimensionless factor, trying to add it with other dimensionless factors."
    )
)
'''--------------------------------------GHR'''
'''a_combined'''
agent.knowledge_pool.add_knowledge(
    Knowledge(
        source="GHR",
        key=r'$v2_dynamic_adjustment$: The symbol in productSpeed unit which represents relative change in speed ' + \
            r'between the ego vehicle and the preceding vehicle, ' + \
            r'$s$: The symbol in distance unit which represents the current following distance',
        target=r'$a_combined$: The symbol in acceleration unit which represents a combined influence ' + \
               'of relative change in speed and distance aspects on acceleration',
        content=
        r"In order to reflect both the relative change in speed $v2_dynamic_adjustment$ and the current following distance $s$, " + \
        r"human experts use $v2_dynamic_adjustment$ as the dividend and $s$ as the divisor to obtain a new symbol #v2_dynamic_adjustment/s#.. " + \
        r"It can represent the impact of relative change in speed and distance aspects. " + \
        r"It can be further condensed as the combined influence on acceleration of the following vehicle, marked as $a_combined$.",
        comment=r"Good symbol that reflects the calculation method of the relative change in speed and distance.",
        reflection=r"It's a good symbol, so I should learn from it to generate combined influence.",
    )
)
'''--------------------------------------Helly'''
'''a_delta_v'''
agent.knowledge_pool.add_knowledge(
    Knowledge(
        source="Helly",
        key=r'$k_1$: The symbol in frequency unit which represents the sensitivity of the following vehicle ' + \
            'to the speed difference between itself and the leading vehicle, ' + \
            "$delta_v$: The symbol in speed unit which represents subtracting the speed of the preceding vehicle " + \
            "from the speed of ego vehicle",
        target=r'$a_delta_v$: The symbol in acceleration unit which represents the influence of ' + \
               r'speed difference factor on acceleration',
        content=
        r"Free coefficient $k_1$ can proportionally adjust the impact of the following vehicle to the speed difference " + \
        r"Human experts multiply the frequency-unit sensitivity coefficient $k_1$ with the speed difference $delta_v$ to quantify the " + \
        r"impact of the speed difference factor on acceleration, which is #k_1*delta_v#. " + \
        r"It can be further condensed as the influence of speed difference factor on acceleration, marked as $a_delta_v$.",
        comment=r"Good symbol that reflects the calculation method of the frequency-unit sensitivity coefficient with the speed difference factor, " + \
                "which is the multiplication of the frequency-unit sensitivity coefficient with the speed difference.",
        reflection=r"I can multiply the frequency-unit sensitivity coefficient with the speed difference.",
    )
)
'''s_difference'''
agent.knowledge_pool.add_knowledge(
    Knowledge(
        source="Helly",
        key=r'$s$: The symbol in distance unit which represents the current following distance, ' + \
            '$s_0$: The symbol in distance unit which represents the desired safety distance between vehicles',
        target=r'$s_difference$: The symbol in distance unit which represents the distance difference between ' + \
               'the current following distance and the desired safety distance between vehicles',
        content=
        r"In order to reflect the subtraction relationship between the current following distance $s$ " + \
        r"and the desired safety distance between vehicles $s_0$, " + \
        r"human experts use $s$ as the minuend and $s_0$ as the subtrahend to obtain a new symbol #s-s_0#. " + \
        r"It can represent the difference between the actual following distance $s$ and the desired following distance $s_0$. " + \
        r"It can be further condensed as the distance difference, marked as $s_difference$.",
        comment=r"Good symbol that reflects the calculation method of the difference between the actual following distance and the desired following distance, which is the subtraction of the desired following distance from the actual following distance.",
        reflection=r"It's a good symbol, so I should learn from it to get distance difference.",
    )
)
'''a_s_difference'''
agent.knowledge_pool.add_knowledge(
    Knowledge(
        source="Helly",
        key=r'$k_2$: The symbol in productFrequency unit which represents the sensitivity of the following vehicle ' + \
            'to the distance difference between itself and the leading vehicle, ' + \
            '$s_difference$: The symbol in distance unit which represents the distance difference between ' + \
            'the current following distance and the desired safety distance between vehicles',
        target=r'$a_s_difference$: The symbol in acceleration unit which represents the influence of ' + \
               r'distance difference factor on acceleration',
        content=
        r"Free coefficient $k_2$ can proportionally adjust the impact of the following vehicle to the distance difference. " + \
        r"Human experts multiply the productFrequency-unit sensitivity coefficient $k_2$ with the distance difference $s_difference$ " + \
        r"to quantify the impact of the distance difference factor on acceleration, which is #k_2*s_difference#. " + \
        r"It can be further condensed as the influence of distance difference factor on acceleration, marked as $a_s_difference$.",
        comment=r"Good symbol that reflects the calculation method of the productFrequency-unit sensitivity coefficient with the distance difference factor, " + \
                "which is the multiplication of the productFrequency-unit sensitivity coefficient with the distance difference.",
        reflection=r"I can multiply the productFrequency-unit sensitivity coefficient with the distance difference to proportionally " + \
                   "adjust the impact of the following vehicle to the distance difference.",
    )
)
'''save_target_names'''
agent.knowledge_pool.save_target_names()
