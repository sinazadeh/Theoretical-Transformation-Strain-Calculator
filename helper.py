import sympy as sp
import numpy as np


class TransformationStrainCalculator:
    def __init__(
        self,
        a0_val,
        a_val,
        b_val,
        c_val,
        beta_val,
        texture_direction,
        deformation_directions,
    ):
        self.a0_val = a0_val
        self.a_val = a_val
        self.b_val = b_val
        self.c_val = c_val
        self.beta_val = np.deg2rad(beta_val)  # Convert degrees to radians
        self.texture_direction = np.array(texture_direction)
        self.deformation_directions = [
            np.array(dir) for dir in deformation_directions]
        self.lattice_constants_sets = [
            ([1, 0, 0], [0, 1, 1], [0, -1, 1]),
            ([-1, 0, 0], [0, -1, -1], [0, -1, 1]),
            ([1, 0, 0], [0, -1, 1], [0, -1, -1]),
            ([-1, 0, 0], [0, 1, -1], [0, -1, -1]),
            ([0, 1, 0], [1, 0, 1], [1, 0, -1]),
            ([0, -1, 0], [-1, 0, -1], [1, 0, -1]),
            ([0, 1, 0], [1, 0, -1], [-1, 0, -1]),
            ([0, -1, 0], [-1, 0, 1], [-1, 0, -1]),
            ([0, 0, 1], [1, 1, 0], [-1, 1, 0]),
            ([0, 0, -1], [-1, -1, 0], [-1, 1, 0]),
            ([0, 0, 1], [-1, 1, 0], [-1, -1, 0]),
            ([0, 0, -1], [1, -1, 0], [-1, -1, 0]),
        ]  # Reference: https://doi.org/10.1016/j.pmatsci.2004.10.001

    def solve_equations(self, a_lc_1, a_lc_2, a_lc_3):
        """
        Solve the equations to determine the transformation matrix.

        Args:
            a_lc_1, a_lc_2, a_lc_3 (list or array-like): Lattice correspondence of austenite.

        Returns:
            sympy.Matrix: A 3x3 matrix representing the transformation matrix.

        References:
            - https://doi.org/10.1016/S1359-6454(02)00123-4
            - https://doi.org/10.1016/j.pmatsci.2004.10.001
        """
        B11, B12, B13, B21, B22, B23, B31, B32, B33 = sp.symbols(
            "B11 B12 B13 B21 B22 B23 B31 B32 B33"
        )

        # Equations based on the transformation and given conditions
        equations = [
            sp.Eq(
                B11 * self.a0_val * a_lc_1[0]
                + B12 * self.a0_val * a_lc_1[1]
                + B13 * self.a0_val * a_lc_1[2],
                self.a_val * sp.cos(self.beta_val),
            ),
            sp.Eq(
                B21 * self.a0_val * a_lc_1[0]
                + B22 * self.a0_val * a_lc_1[1]
                + B23 * self.a0_val * a_lc_1[2],
                self.a_val * sp.sin(self.beta_val),
            ),
            sp.Eq(
                B31 * self.a0_val * a_lc_1[0]
                + B32 * self.a0_val * a_lc_1[1]
                + B33 * self.a0_val * a_lc_1[2],
                0,
            ),
            sp.Eq(
                B11 * self.a0_val * a_lc_2[0]
                + B12 * self.a0_val * a_lc_2[1]
                + B13 * self.a0_val * a_lc_2[2],
                0,
            ),
            sp.Eq(
                B21 * self.a0_val * a_lc_2[0]
                + B22 * self.a0_val * a_lc_2[1]
                + B23 * self.a0_val * a_lc_2[2],
                0,
            ),
            sp.Eq(
                B31 * self.a0_val * a_lc_2[0]
                + B32 * self.a0_val * a_lc_2[1]
                + B33 * self.a0_val * a_lc_2[2],
                self.b_val,
            ),
            sp.Eq(
                B11 * self.a0_val * a_lc_3[0]
                + B12 * self.a0_val * a_lc_3[1]
                + B13 * self.a0_val * a_lc_3[2],
                self.c_val,
            ),
            sp.Eq(
                B21 * self.a0_val * a_lc_3[0]
                + B22 * self.a0_val * a_lc_3[1]
                + B23 * self.a0_val * a_lc_3[2],
                0,
            ),
            sp.Eq(
                B31 * self.a0_val * a_lc_3[0]
                + B32 * self.a0_val * a_lc_3[1]
                + B33 * self.a0_val * a_lc_3[2],
                0,
            ),
        ]

        # Solve the system of equations
        solutions = sp.solve(
            equations, (B11, B12, B13, B21, B22, B23, B31, B32, B33))

        # Create the solution matrix
        solution_matrix = sp.Matrix(
            [
                [solutions[B11], solutions[B12], solutions[B13]],
                [solutions[B21], solutions[B22], solutions[B23]],
                [solutions[B31], solutions[B32], solutions[B33]],
            ]
        )

        return solution_matrix

    @staticmethod
    def symbolic_matrix_to_numpy(sym_matrix):
        return np.array(sym_matrix).astype(np.float64)

    @staticmethod
    def calculate_transformation_strain(deformation_direction, B_np):
        """
        Calculate the transformation strain as a percentage based on a given deformation direction and transformation matrix.

        Args:
            deformation_direction (numpy.ndarray): A vector representing the direction of deformation. Should be a 1D numpy array or a similar iterable data structure.
            transformation_matrix (numpy.ndarray): A matrix representing the transformation. Should be a 2D numpy array.

        Returns:
            float: The transformation strain expressed as a percentage. This value quantifies the extent of strain induced in a direction due to the deformation, scaled as a percentage of the original dimensions.

        Reference:
            - http://dx.doi.org/10.1016/j.actamat.2015.03.022
        """
        deformation_direction_T = deformation_direction.reshape(
            1, -1
        )  # Transpose of deformation_direction for matrix multiplication
        BT = B_np.T  # Transpose of B
        numerator = np.sqrt(
            np.dot(
                np.dot(deformation_direction_T, np.dot(
                    BT, B_np)), deformation_direction
            )[0]
        )
        denominator = np.sqrt(
            np.dot(deformation_direction_T, deformation_direction)[0])
        Eps = ((numerator / denominator) - 1) * 100
        return Eps

    def print_results(self):
        results = []
        # Initialize with a very small number
        max_transformation_strain = -float("inf")
        associated_direction = None
        for deformation_direction in self.deformation_directions:
            print(
                f"\nTexture direction: {self.texture_direction}"
                + f", Deformation direction: {deformation_direction}"
            )

            # Calculate the dot product
            dot_product = np.dot(self.texture_direction, deformation_direction)

            # Calculate the magnitude of each vector
            magnitude_a = np.linalg.norm(self.texture_direction)
            magnitude_b = np.linalg.norm(deformation_direction)

            # Ensure no division by zero or floating point inaccuracies
            if np.isclose(magnitude_a * magnitude_b, 0) or np.isclose(
                dot_product / (magnitude_a * magnitude_b), 1.0
            ):
                angle_radians = (
                    0  # The vectors are parallel or one of them is a zero vector
                )
            else:
                # Calculate the angle in radians
                cos_theta = dot_product / (magnitude_a * magnitude_b)
                cos_theta_clipped = np.clip(
                    cos_theta, -1, 1
                )  # Ensure the value is within the domain of arccos
                angle_radians = np.arccos(cos_theta_clipped)

            # Convert the angle to degrees
            angle_degrees = np.degrees(angle_radians)

            print(
                "Angle between texture direction and deformation direction: {:.2f} degrees".format(
                    angle_degrees
                )
            )

            # Initialize list to store Eps values for this deformation direction
            Eps_values = []

            # Calculate Eps for each set of lattice constants for the current deformation direction
            for a_lc_1, a_lc_2, a_lc_3 in self.lattice_constants_sets:
                solution_matrix = self.solve_equations(a_lc_1, a_lc_2, a_lc_3)
                B_np = self.symbolic_matrix_to_numpy(
                    solution_matrix
                )  # Convert to numpy array
                Eps = self.calculate_transformation_strain(
                    deformation_direction, B_np)
                Eps_values.append(Eps)

            # Print Eps values for each variation
            max_Eps = max(Eps_values)  # Find the maximum Eps value
            if max_Eps > max_transformation_strain:
                max_transformation_strain = max_Eps
                associated_direction = deformation_direction
                associated_degree = angle_degrees

            # Print Eps values for each variation with highlighting for the maximum value
            variation_number = 1
            for i, Eps in enumerate(Eps_values):
                # Determine if the current variation should have a prime appended
                prime = "'" if i % 2 else ""

                # Check if the current Eps value is the maximum
                if Eps == max_Eps:
                    print(f"V{variation_number}{prime}: {Eps:.5f}% <- Highest!")
                else:
                    print(f"V{variation_number}{prime}: {Eps:.5f}%")

                # Only increment the variation number every two iterations (for each pair)
                if i % 2:
                    variation_number += 1

            results.append(
                (deformation_direction, Eps_values)
            )  # Store deformation direction and Eps values for each direction
        print("\nResults:")
        print(
            f"Maximum Theoretical Transformation Strain: {max_transformation_strain:.5f}%")
        print(
            f"Deformation Direction: {associated_direction}, Texture Direction: {self.texture_direction}, Degree: {associated_degree:.2f}"
        )

        # return results
