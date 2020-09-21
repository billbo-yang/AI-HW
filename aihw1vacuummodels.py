from agents import ReflexVacuumAgent, TrivialVacuumEnvironment

def Model(start, size, status=[]):
    print("---Start: {}, Order: {}---".format(start, *status))
    agent = ReflexVacuumAgent()
    env = TrivialVacuumEnvironment(size)
    env.add_thing(agent, start)

    if(len(status) > 0):
        for i in range(size):
            env.status[(i, 0)] = status[i]

    clean = False
    while not clean:
        clean = True
        for key in env.status:
            if env.status[key] == "Dirty": clean = False

        if not clean: env.step()

    return env.agents[0].performance

def main():
    print("-----Steps Taken-----")
    aabc = Model((0, 0), 2, ['Clean', 'Clean'])
    aadbc = Model((0, 0), 2, ['Dirty', 'Clean'])
    aacbd = Model((0, 0), 2, ['Clean', 'Dirty'])
    aabd = Model((0, 0), 2, ['Dirty', 'Dirty'])
    aavg = (aabc+aadbc+aacbd+aabd)/4

    babc = Model((1, 0), 2, ['Clean', 'Clean'])
    badbc = Model((1, 0), 2, ['Dirty', 'Clean'])
    bacbd = Model((1, 0), 2, ['Clean', 'Dirty'])
    babd = Model((1, 0), 2, ['Dirty', 'Dirty'])
    bavg = (babc+badbc+bacbd+babd)/4

    tavg = (aavg+bavg)/2
    print("-----Performance, higher is better*-----")
    print("*except when environment starts as clean")
    print("A Start, ABClean:\t{}".format(aabc))
    print("A Start, ADirtBClean:\t{}".format(aadbc))
    print("A Start, ACleanBDirt:\t{}".format(aacbd))
    print("A Start, ABDirt:\t{}".format(aabd))
    print("\n")
    print("B Start, ABClean:\t{}".format(babc))
    print("B Start, ADirtBClean:\t{}".format(badbc))
    print("B Start, ACleanBDirt:\t{}".format(bacbd))
    print("B Start, ABDirt:\t{}".format(babd))
    print("\n")
    print("Total Average:\t\t{}".format(tavg))

if __name__ == "__main__":
    main()