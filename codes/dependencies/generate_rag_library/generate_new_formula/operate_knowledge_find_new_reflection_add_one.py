import os
import sys
sys.path.append(os.path.abspath(os.getcwd()))
from codes.trafficSR.D_updation_by_LLM.RAG_Agent import RAG_AGENT
from codes.trafficSR.D_updation_by_LLM.Modules.Knowledge_Pool import Knowledge

port = "7890"
agent = RAG_AGENT(port=port, memory_path="../../../ragLibrary/memory_ngsim_find_new_reflection")
knowledge_length = len(agent.knowledge_pool.content._collection.get(include=['embeddings'])['embeddings'])
'''--------------------------------------IDM'''
'''factor_v_quadratic'''
agent.knowledge_pool.add_knowledge(
    Knowledge(
        source="IDM",
        key=r'$factor_v_squared$: The symbol in dimensionless unit which represents the square of the speed ratio influence factor',
        target=r'$factor_v_quadratic$: The symbol in dimensionless unit which represents the quadratic of the speed ratio influence factor',
        content=
        r"In the vehicle following formula, human experts often square the dimensionless unit variable to achieve a more accurate fitting effect. " + \
        r"Human experts square $factor_v_squared$ to obtain a new symbol $factor_v_quadratic$, which is #n2(factor_v_squared)#. " + \
        r"It can be further condensed as the influencing factor of speed, marked as $factor_v_quadratic$.",
        comment=r"When there are dimensionless symbols, operations such as square or root can be performed to obtain " + \
                r"more accurate symbols that are also dimensionless.",
        reflection=r"I will remember when encountering dimensionless symbols, trying to square or root them."
    )
)
'''factor_integrated'''
agent.knowledge_pool.add_knowledge(
    Knowledge(
        source="IDM",
        key=r'$factor_v_squared$: The symbol in dimensionless unit which represents the square of the speed influence factor, ' + \
            r'$factor_s_squared$: The symbol in dimensionless unit which represents the square of the distance influence factor',
        target=r'$factor_integrated$: The symbol in dimensionless unit which represents both speed and distance influence factors',
        content=
        r"In the vehicle following formula, human experts often add dimensionless factor with other dimensionless " + \
        r"factor to comprehensively reflect the influence of multiple factors. " + \
        r"Human experts add $factor_v_squared$ and $factor_s_squared$ to obtain the impact of both aspects, which is #factor_v_squared+factor_s_squared#. " + \
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
