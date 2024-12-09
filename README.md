**Step-by-Step Solution**

**Given:**

- Beam span: 18 ft (simply supported at both ends)
- A concentrated load of 3000 lbs at 4 ft from the left support.
- A uniformly distributed load (UDL) of 400 lbs/ft from 4 ft to 18 ft. Thus, the UDL covers 14 ft (from x = 4 ft to x = 18 ft).

**Loads:**

1. **Concentrated load**: \( P = 3000 \, \text{lbs} \) at \( x = 4 \, \text{ft} \).
2. **Uniform load**: \( w = 400 \, \text{lbs/ft} \) over a length of 14 ft (from 4 ft to 18 ft).
   - Total uniform load: \( W = w \times 14 = 400 \times 14 = 5600 \, \text{lbs} \).

**Reactions:**

Let:
- \( R_1 \) = reaction at left support (x=0).
- \( R_2 \) = reaction at right support (x=18 ft).

Equilibrium equations:

1. **Sum of vertical forces = 0:**
   \[
   R_1 + R_2 - 3000 - 5600 = 0
   \]
   \[
   R_1 + R_2 = 8600 \, \text{lbs}
   \]

2. **Sum of moments about the left support = 0:**

   Taking moments about the left end (counterclockwise positive):

   - The 3000 lb load acts at 4 ft: Moment = \( 3000 \times 4 = 12000 \, \text{ft-lbs} \) (clockwise).
   - The UDL of 5600 lbs acts over 14 ft starting at 4 ft. Its centroid is at the midpoint of that segment, i.e., at \( 4 + \frac{14}{2} = 4 + 7 = 11 \, \text{ft} \) from the left support. Moment due to UDL = \( 5600 \, \text{lbs} \times 11 \, \text{ft} = 61600 \, \text{ft-lbs} \) (clockwise).
   - \( R_2 \) acts upward at 18 ft: Moment = \( R_2 \times 18 \, \text{ft} \) (counterclockwise).

   Summation of moments about left support:
   \[
   R_2 \cdot 18 - 3000 \cdot 4 - 5600 \cdot 11 = 0
   \]
   \[
   18 R_2 - 12000 - 61600 = 0
   \]
   \[
   18 R_2 = 73600
   \]
   \[
   R_2 = \frac{73600}{18} \approx 4089 \, \text{lbs}
   \]

3. **Find \( R_1 \):**
   \[
   R_1 = 8600 - R_2 = 8600 - 4089 = 4511 \, \text{lbs}
   \]

**Reactions:**
- \( R_1 \approx 4511 \, \text{lbs} \)
- \( R_2 \approx 4089 \, \text{lbs} \)

---

**Shear Diagram Construction:**

- Start from the left end:
  At x=0, shear \( V(0^+) = R_1 = 4511 \, \text{lbs} \).

- Move to x=4 ft (just before the 3000 lb load):
  No distributed load up to 4 ft, so shear remains constant:
  \( V(4^-)=4511 \, \text{lbs} \).

- At x=4 ft, the 3000 lb load acts downward:
  Just to the right of the load:
  \[
  V(4^+) = 4511 - 3000 = 1511 \, \text{lbs}
  \]

- From x=4 ft to x=18 ft, there is a uniform load of 400 lbs/ft:
  As we move to the right, the shear decreases linearly due to this distributed load.

  At x=18 ft (just before the right support):
  The total downward load from 4 ft to 18 ft is 5600 lbs. Reducing the shear from 1511 lbs:
  \[
  V(18^-)=1511 - 5600 = -4089 \, \text{lbs}
  \]

- At the right support (x=18 ft), \( R_2 = 4089 \, \text{lbs} \) acts upward, bringing the final shear back to zero:
  \[
  V(18^+) = -4089 + 4089 = 0
  \]

**Key Shear Values:**
- Maximum positive shear = 4511 lbs (at the left support).
- Minimum shear = -4089 lbs (just before the right support).

---

**Bending Moment Diagram:**

- At x=0 (left support), \( M=0 \).

- From 0 to 4 ft (no load except supports):
  Moment increases linearly:
  \[
  M(x) = R_1 x = 4511 x
  \]
  At x=4 ft:
  \[
  M(4) = 4511 \times 4 = 18044 \, \text{ft-lbs}
  \]

- From 4 ft to 18 ft, the bending moment is influenced by both the point load and the uniform load.

  One way to find the maximum moment is to write the moment equation or to use the shear diagram integral:

  After x=4 ft:
  Initial moment at x=4 ft: \( M(4)=18044 \, \text{ft-lbs} \).

  The shear at x=4 ft is now 1511 lbs. As we move from 4 to x:
  \[
  V(t) = 1511 - 400(t-4), \quad \text{for } 4 \le t \le 18.
  \]

  The moment is the integral of shear:
  \[
  M(x) = M(4) + \int_{4}^{x} V(t) dt
  = 18044 + \int_{4}^{x} [1511 - 400(t-4)] dt.
  \]

  Let \( y = x-4 \):
  \[
  M(x) = 18044 + \int_{0}^{x-4} [1511 - 400y] dy
  \]
  \[
  M(x) = 18044 + [1511y - 200y^2]_{0}^{x-4}
  = 18044 + 1511(x-4) - 200(x-4)^2.
  \]

  To find the maximum moment, differentiate w.r.t x:
  \[
  \frac{dM}{dx} = 1511 - 400(x-4).
  \]
  Set \(\frac{dM}{dx}=0\) for max/min:
  \[
  1511 - 400(x-4) = 0
  \]
  \[
  400(x-4) = 1511
  \]
  \[
  x-4 = \frac{1511}{400} \approx 3.7775
  \]
  \[
  x \approx 7.78 \, \text{ft}
  \]

  Substitute \( x = 7.78 \) ft into M(x):
  \[
  M(7.78) \approx 18044 + 1511(3.78) - 200(3.78^2).
  \]

  Compute:
  - \( 1511 \times 3.78 \approx 5712 \)
  - \( (3.78)^2 \approx 14.3 \)
  - \( 200 \times 14.3 = 2860 \)

  Thus:
  \[
  M(7.78) \approx 18044 + 5712 - 2860 = 18044 + 2852 = 20896 \, \text{ft-lbs (approx)}
  \]

  This is the maximum bending moment.

- At the right support (x=18 ft), \( M(18)=0 \).

**Maximum Bending Moment:**
\[
M_{\text{max}} \approx 20,900 \, \text{ft-lbs (rounded)}
\]

---

**Summary of Results:**

- **Reactions:**
  - \( R_1 \approx 4511 \, \text{lbs} \)
  - \( R_2 \approx 4089 \, \text{lbs} \)

- **Maximum Shear:**
  - \( V_{\max} = 4511 \, \text{lbs} \) at the left support.

- **Maximum Bending Moment:**
  - \( M_{\max} \approx 20,900 \, \text{ft-lbs} \) at approximately \( x = 7.8 \, \text{ft} \) from the left support.

---

**Shear Diagram (Qualitative):**
- Starts at +4511 lbs at x=0.
- Remains constant until x=4 ft.
- Drops by 3000 lbs at x=4 ft to 1511 lbs.
- Then linearly decreases from 1511 lbs at x=4 ft to -4089 lbs at x=18 ft due to the 400 lbs/ft load.
- Jumps back up by 4089 lbs at x=18 ft to 0.

**Moment Diagram (Qualitative):**
- Starts at 0 at x=0.
- Increases linearly up to x=4 ft to 18044 ft-lbs.
- Then follows a parabolic curve due to the uniform load, reaching a maximum around x=7.8 ft (~20900 ft-lbs).
- Decreases back down to 0 at x=18 ft.

These diagrams and values provide the complete internal force distribution for the given loaded beam.
