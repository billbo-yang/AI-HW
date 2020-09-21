from agents import ReflexVacuumAgent, TrivialVacuumEnvironment

def Model(start, a, b):
    print("---Start: {}, (0,0):{} (1,0):{}---".format(start, a, b))
    agent = ReflexVacuumAgent()
    env = TrivialVacuumEnvironment()
    env.add_thing(agent, start)
    env.status = {(1,0): b, (0,0) : a}

    while(env.status != {(1,0):'Clean' , (0,0) : 'Clean'}):
        env.step()

    return env.agents[0].performance

def main():
    print("-----Steps Taken-----")
    aabc = Model((0, 0), 'Clean', 'Clean')
    aadbc = Model((0, 0), 'Dirty', 'Clean')
    aacbd = Model((0, 0), 'Clean', 'Dirty')
    aabd = Model((0, 0), 'Dirty', 'Dirty')
    aavg = (aabc+aadbc+aacbd+aabd)/4

    babc = Model((1, 0), 'Clean', 'Clean')
    badbc = Model((1, 0), 'Dirty', 'Clean')
    bacbd = Model((1, 0), 'Clean', 'Dirty')
    babd = Model((1, 0), 'Dirty', 'Dirty')
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