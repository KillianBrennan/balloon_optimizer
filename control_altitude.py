import numpy as np

# balloon constants
BALLOON_VOLUME_INITIAL = 1050  # m^3
BALLOON_VOLUME_MAX = 1100  # m^3
BALAST_MASS_INITIAL = 300  # kg
EMPTY_MASS = 500  # kg
DRAG_COEFFICIENT = 0.47  # Approximate for a sphere
TAKE_OFF_ALTITUDE = 500  # m above sea level
LANDING_ALTITUDE = 500  # m above sea level

# physical constants
g = 9.81  # m/s^2 (gravitational acceleration)
R_air = 287.058  # J/(kg·K) (specific gas constant for air)
P0 = 101325  # Pa (sea level pressure)
T0 = 288.15  # K (sea level temperature)
L = -0.0065  # K/m (temperature lapse rate)
R_hydrogen = 4124.2  # J/(kg·K) (specific gas constant for hydrogen)

# numerical parameters
MODEL_TIME_STEP = 10  # seconds
OUTPUT_TIME_STEP = 60  # seconds (1 hour)
TOTAL_TIME = 10 * 3600  # seconds (100 hours)


def main():

    return


def model_altitude(balast_events, venting_events):
    """
    Model the altitude of the balloon over time given balast and venting events.
    balast_events: list of tuples (time, mass) where mass is the amount of balast released at that time
    venting_events: list of tuples (time, mass) where mass is the amount of gas vented at that time
    """
    times = []
    altitudes = []
    gas_masses = []

    time = 0
    altitude = TAKE_OFF_ALTITUDE
    balast_mass = BALAST_MASS_INITIAL
    balloon_volume = BALLOON_VOLUME_INITIAL
    gas_mass = balloon_volume * compute_balloon_density(TAKE_OFF_ALTITUDE)
    balast_index = 0
    venting_index = 0

    while time < TOTAL_TIME:
        if altitude < LANDING_ALTITUDE:
            altitude = LANDING_ALTITUDE
        else:
            # Check for balast drop events
            if balast_index < len(balast_events) and time >= balast_events[balast_index][0]:
                balast_mass -= balast_events[balast_index][1]
                balast_index += 1

            # Check for venting events
            if (
                venting_index < len(venting_events)
                and time >= venting_events[venting_index][0]
            ):
                gas_mass -= venting_events[venting_index][1]
                venting_index += 1

            # update balloon volume based on gas mass and density
            balloon_volume = gas_mass / compute_balloon_density(altitude)

            # compute vertical velocity and update altitude
            velocity = compute_terminal_velocity(
                altitude, balloon_volume, EMPTY_MASS + balast_mass + gas_mass
            )
            altitude += velocity * MODEL_TIME_STEP
            # vent gas if it expands beyond the maxballoon volume
            if balloon_volume > BALLOON_VOLUME_MAX:
                gas_mass = compute_balloon_density(altitude) * BALLOON_VOLUME_MAX

        # Update times and altitudes arrays
        if time % OUTPUT_TIME_STEP == 0:
            times.append(time)
            altitudes.append(altitude)
            gas_masses.append(gas_mass)

        time += MODEL_TIME_STEP

    times = np.array(times)
    altitudes = np.array(altitudes)
    gas_masses = np.array(gas_masses)

    return times, altitudes, gas_masses


def compute_terminal_velocity(altitude, balloon_volume, gross_mass):
    """
    Compute terminal velocity of the balloon at a given altitude, volume, and gross mass.
    """
    buoyant_force = compute_buoyant_force(altitude, balloon_volume)
    total_mass = balloon_volume * compute_balloon_density(altitude) + gross_mass
    total_force = buoyant_force - total_mass * g
    force_sign = np.sign(total_force)
    total_force = np.abs(total_force)

    baloon_radius = (3 * balloon_volume / (4 * np.pi)) ** (1 / 3)
    cross_sectional_area = np.pi * baloon_radius**2
    terminal_velocity = np.sqrt(
        2
        * total_force
        / (DRAG_COEFFICIENT * compute_ambient_density(altitude) * cross_sectional_area)
    )
    terminal_velocity *= force_sign
    return terminal_velocity


def compute_ambient_density(altitude):
    """
    Compute ambient air density at a given altitude.
    """
    pressure = pressure_from_altitude(altitude)
    temperature = T0 - L * altitude  # Simplified temperature model

    density = pressure / (R_air * temperature)
    return density


def compute_balloon_density(altitude, temperature=None):
    """
    Compute balloon density at a given altitude.
    """
    pressure = pressure_from_altitude(altitude)
    if temperature is None:
        temperature = T0 - L * altitude  # Simplified temperature model

    density = pressure / (R_hydrogen * temperature)
    return density


def pressure_from_altitude(altitude):
    """
    Compute pressure from altitude using standard atmosphere model.
    """

    T = T0 + L * altitude
    P = P0 * (T / T0) ** (-g / (L * R_air))
    return P


def compute_buoyant_force(altitude, balloon_volume):
    """
    Compute buoyant force
    """

    buoyant_force = (
        (compute_ambient_density(altitude) - compute_balloon_density(altitude))
        * g
        * balloon_volume
    )

    return buoyant_force


def apply_radiation(lat, lon, altitude, time):

    return


if __name__ == "__main__":
    main()
