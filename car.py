from constants import fuel_dict, transmission_dict

class Car:
    def __init__(self, make, model, year, fuel_type, cylinders, displacement, transmission, drive, city_lkm, highway_lkm, combination_lkm, car_class):
        self.make = make
        self.model = model
        self.year = year
        self.fuel_type = fuel_type
        self.cylinders = cylinders
        self.displacement = displacement
        self.transmission = transmission
        self.drive = drive
        self.city_lkm = city_lkm
        self.highway_lkm = highway_lkm
        self.combination_lkm = combination_lkm
        self.car_class = car_class

    def __str__(self):
        fuel = fuel_dict[self.fuel_type]
        transmission = transmission_dict[self.transmission]

        string = f'{self.make} {self.model} {self.year}, {fuel}, {self.cylinders} cilindros, {self.displacement} L, {transmission}, {self.drive}, {self.car_class}'

        return string
    
    def to_dict(self):
        return {
            'make': self.make,
            'model': self.model,
            'year': self.year,
            'fuel_type': self.fuel_type,
            'cylinders': self.cylinders,
            'displacement': self.displacement,
            'transmission': self.transmission,
            'drive': self.drive,
            'city_lkm': self.city_lkm,
            'highway_lkm': self.highway_lkm,
            'combination_lkm': self.combination_lkm,
            'car_class': self.car_class
        }
    
    @staticmethod
    def from_neo4j(db_make, db_car, db_powertrain, db_car_class):
        cylinders = 0 if db_powertrain['cylinders'] == 'N/A' else db_powertrain['cylinders']
        displacement = 0 if db_powertrain['displacement'] == 'N/A' else db_powertrain['displacement']

        return Car(
            db_make['make'],
            db_car['model'],
            db_car['year'],
            db_powertrain['fuel_type'],
            cylinders,
            displacement,
            db_powertrain['transmission'],
            db_powertrain['drive'],
            db_car['city_lkm'],
            db_car['highway_lkm'],
            db_car['combination_lkm'],
            db_car_class['car_class']
        )

class Powertrain:
    def __init__(self, cylinders, displacement, fuel_type, drive, transmission):
        self.cylinders = cylinders
        self.displacement = displacement
        self.fuel_type = fuel_type
        self.drive = drive
        self.transmission = transmission

    def __str__(self):
        fuel = fuel_dict[self.fuel_type]
        transmission = transmission_dict[self.transmission]
        
        string = f'{self.cylinders} cilindros, {self.displacement} L, {fuel}, {self.drive}, {transmission}'

        return string

    def __repr__(self):
        return f'Powertrain({self.cylinders}, {self.displacement}, {self.fuel_type}, {self.drive}, {self.transmission})'
    
    def from_dict(data):
        return Powertrain(
            data['cylinders'],
            data['displacement'],
            data['fuel_type'],
            data['drive'],
            data['transmission']
        )