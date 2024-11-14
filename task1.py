import pulp

model = pulp.LpProblem("Production_Optimization", pulp.LpMaximize)

x = pulp.LpVariable("Lemonade", lowBound=0, cat="Continuous")
y = pulp.LpVariable(
    "FruitJuice", lowBound=0, cat="Continuous"
)

model += 2 * x + y <= 100, "Water"
model += 1 * x <= 50, "Sugar"
model += 1 * x <= 30, "LemonJuice"
model += 2 * y <= 40, "FruitPuree"

model += x + y, "TotalProducts"

model.solve()

print(f"Status: {pulp.LpStatus[model.status]}")
print(f"Lemonade: {pulp.value(x)}")
print(f"FruitJuice: {pulp.value(y)}")
print(f"Total Products: {pulp.value(model.objective)}")
