Now I want to teach you how to use existing symbols in the table to create new combinations.
Firstly, operators need to be used to connect variables and constants. We don't want the same symbols on both sides of
the operator.

The left and right sides of the add and sub signs must be in the same unit. For example, $s$ and $v$ cannot add or
subtract from each other, because they have units [1,0] and [1,-1] respectively.
For example, physical_units of token $s$ and $s_0$ are [1,0], the physical units of $s+s_0$, $s_0+s$, $s-s_0$, $s_0-s$
are all [1,0].And physical_units of token $v$ and $v_0$ are [1,-1], the physical units of $v+v_0$, $v_0+v$, $v-v_0$,
$v_0-v$ are
all [1,-1].

The physical units of the mul result is the sum of the physical units around the multiplication sign, for example, the
physical units of $s * s_0$ is [2,0], the physical units of $v * s$ is [2,-1].
The physical units of the div result is the subtraction of the physical units around the division sign, for example, the
physical units of $s/s_0$ is [1,0]-[1,0]=[0,0], the physical units of $v/s$ is [1,-1]-[1,0]=[0,-1].

n2 represents square, and the physical units of $s^2$ is twice that of s, which is [2,0]. The physical units of $v_0^2$
is twice that of $v_0$, which is [2,-2].

Therefore, if I need a combination with units of [0,0], I can obtain it from $s_0/s$, $v/v_0$. Please be sure to pay
attention to the units when combining.