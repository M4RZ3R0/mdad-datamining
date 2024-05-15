from neo4j import GraphDatabase

class Neo4JDriver:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def get_cars(self):
        with self.driver.session() as session:
            return session.read_transaction(self._get_cars)
        
    @staticmethod
    def _get_cars(tx):
        query = (
            """MATCH (m:Make)-[:MAKES]-(c:Car)-[:HAS_POWERTRAIN]->(p:Powertrain)
            MATCH (c)-[:IS_A]-(cc:CarClass)
            RETURN m, c, p, cc"""
        )

        result = tx.run(query)
        return [
            (record['m'], record['c'], record['p'], record['cc']) for record in result
        ]