            int totalUpdateCandidates = 0;
            for (size_t i = 0; i < currentUpdateCandidatesPerTask.size(); ++i)
            {
                currentUpdateCandidatesOffset.push_back(totalUpdateCandidates);
                totalUpdateCandidates += currentUpdateCandidatesPerTask.at(i);
            }
            procUpdateCandidate.resize(totalUpdateCandidates, 0);
            indexStructure.resize(totalUpdateCandidates, 0);
            indexStructureGlobal.resize(totalUpdateCandidates, 0);
            indexAtom.resize(totalUpdateCandidates, 0);
            indexCoordinate.resize(totalUpdateCandidates, 0);
            // Increase size of this vectors (only rank 0).
            currentRmseFraction.resize(totalUpdateCandidates, 0.0);
            thresholdLoopCount.resize(totalUpdateCandidates, 0.0);
        }
        else
        {
            procUpdateCandidate.resize(myCurrentUpdateCandidates, 0);
            indexStructure.resize(myCurrentUpdateCandidates, 0);
            indexStructureGlobal.resize(myCurrentUpdateCandidates, 0);
            indexAtom.resize(myCurrentUpdateCandidates, 0);
            indexCoordinate.resize(myCurrentUpdateCandidates, 0);
        }
        for (int i = 0; i < myCurrentUpdateCandidates; ++i)
        {
            procUpdateCandidate.at(i) = myRank;
            UpdateCandidate& c = *(currentUpdateCandidates.at(i));
            indexStructure.at(i) = c.s;
            indexStructureGlobal.at(i) = structures.at(c.s).index;
            indexAtom.at(i) = c.a;
            indexCoordinate.at(i) = c.c;
        }
        if (myRank == 0)
        {
        }
        else
        {
            MPI_Gatherv(&(currentRmseFraction.front()) , myCurrentUpdateCandidates, MPI_DOUBLE, NULL, NULL, NULL, MPI_DOUBLE, 0, comm);
            MPI_Gatherv(&(thresholdLoopCount.front())  , myCurrentUpdateCandidates, MPI_SIZE_T, NULL, NULL, NULL, MPI_SIZE_T, 0, comm);
            MPI_Gatherv(&(procUpdateCandidate.front()) , myCurrentUpdateCandidates, MPI_INT   , NULL, NULL, NULL, MPI_INT   , 0, comm);
            MPI_Gatherv(&(indexStructure.front())      , myCurrentUpdateCandidates, MPI_SIZE_T, NULL, NULL, NULL, MPI_SIZE_T, 0, comm);
            MPI_Gatherv(&(indexStructureGlobal.front()), myCurrentUpdateCandidates, MPI_SIZE_T, NULL, NULL, NULL, MPI_SIZE_T, 0, comm);
            MPI_Gatherv(&(indexAtom.front())           , myCurrentUpdateCandidates, MPI_SIZE_T, NULL, NULL, NULL, MPI_SIZE_T, 0, comm);
            MPI_Gatherv(&(indexCoordinate.front())     , myCurrentUpdateCandidates, MPI_SIZE_T, NULL, NULL, NULL, MPI_SIZE_T, 0, comm);
        }

        if (myRank == 0)
        {
            for (size_t i = 0; i < procUpdateCandidate.size(); ++i)
            {
                if (k == "energy")
                {
                    addTrainingLogEntry(procUpdateCandidate.at(i),
                                        thresholdLoopCount.at(i),
                                        currentRmseFraction.at(i),
                                        indexStructureGlobal.at(i),
                                        indexStructure.at(i));
                }
                else if (k == "force")
                {
                    addTrainingLogEntry(procUpdateCandidate.at(i),
                                        thresholdLoopCount.at(i),
                                        currentRmseFraction.at(i),
                                        indexStructureGlobal.at(i),
                                        indexStructure.at(i),
                                        indexAtom.at(i),
                                        indexCoordinate.at(i));
                }
                else if (k == "charge")
                {
                    addTrainingLogEntry(procUpdateCandidate.at(i),
                                        thresholdLoopCount.at(i),
                                        currentRmseFraction.at(i),
                                        indexStructureGlobal.at(i),
                                        indexStructure.at(i),
                                        indexAtom.at(i));
                }
            }
        }
    }
    sw[k + "_log"].stop();
    sw[k].stop();

    return;
}

double Training::getSingleWeight(size_t element, size_t index)
{
    getWeights();

    return weights.at(element).at(index);
}

void Training::setSingleWeight(size_t element, size_t index, double value)
{
    weights.at(element).at(index) = value;
    setWeights();

    return;
}

vector<
vector<double>> Training::calculateWeightDerivatives(Structure* structure)
{
    Structure& s = *structure;
#ifdef N2P2_NO_SF_GROUPS
    calculateSymmetryFunctions(s, false);
#else
    calculateSymmetryFunctionGroups(s, false);
#endif

    vector<vector<double> > dEdc;
    vector<vector<double> > dedc;
    dEdc.resize(numElements);
    dedc.resize(numElements);
    for (size_t i = 0; i < numElements; ++i)
    {
        size_t n = elements.at(i).neuralNetworks.at("short")
                   .getNumConnections();
        dEdc.at(i).resize(n, 0.0);
        dedc.at(i).resize(n, 0.0);
    }
    for (vector<Atom>::iterator it = s.atoms.begin();
         it != s.atoms.end(); ++it)
    {
        size_t i = it->element;
        NeuralNetwork& nn = elements.at(i).neuralNetworks.at("short");
        nn.setInput(&((it->G).front()));
        nn.propagate();
        nn.getOutput(&(it->energy));
        nn.calculateDEdc(&(dedc.at(i).front()));
        for (size_t j = 0; j < dedc.at(i).size(); ++j)
        {
            dEdc.at(i).at(j) += dedc.at(i).at(j);
        }
    }

    return dEdc;
}

// Doxygen requires namespace prefix for arguments...
vector<
vector<double>> Training::calculateWeightDerivatives(Structure*  structure,
                                                      std::size_t atom,
                                                      std::size_t component)
{
    Structure& s = *structure;
#ifdef N2P2_NO_SF_GROUPS
    calculateSymmetryFunctions(s, true);
#else
    calculateSymmetryFunctionGroups(s, true);
#endif

    vector<vector<double> > dFdc;
    vector<vector<double> > dfdc;
    dFdc.resize(numElements);
    dfdc.resize(numElements);
    for (size_t i = 0; i < numElements; ++i)
    {
        size_t n = elements.at(i).neuralNetworks.at("short")
                   .getNumConnections();
        dFdc.at(i).resize(n, 0.0);
        dfdc.at(i).resize(n, 0.0);
    }
    for (vector<Atom>::iterator it = s.atoms.begin();
         it != s.atoms.end(); ++it)
    {
#ifndef N2P2_FULL_SFD_MEMORY
        collectDGdxia((*it), atom, component);
#else
        it->collectDGdxia(atom, component);
#endif
        size_t i = it->element;
        NeuralNetwork& nn = elements.at(i).neuralNetworks.at("short");
        nn.setInput(&((it->G).front()));
        nn.propagate();
        nn.getOutput(&(it->energy));
#ifndef N2P2_FULL_SFD_MEMORY
        nn.calculateDFdc(&(dfdc.at(i).front()), &(dGdxia.front()));
#else
        nn.calculateDFdc(&(dfdc.at(i).front()), &(it->dGdxia.front()));
#endif
        for (size_t j = 0; j < dfdc.at(i).size(); ++j)
        {
            dFdc.at(i).at(j) += dfdc.at(i).at(j);
        }
    }

    return dFdc;
}

void Training::setTrainingLogFileName(string fileName)
{
    trainingLogFileName = fileName;

    return;
}

size_t Training::getNumConnections(string id) const
{
    size_t n = 0;
    for (auto const& e : elements)
    {
        n += e.neuralNetworks.at(id).getNumConnections();
    }

    return n;
}

vector<size_t> Training::getNumConnectionsPerElement(string id) const
{
    vector<size_t> npe;
    for (auto const& e : elements)
    {
        npe.push_back(e.neuralNetworks.at(id).getNumConnections());
    }

    return npe;
}

vector<size_t> Training::getConnectionOffsets(string id) const
{
    vector<size_t> offset;
    size_t n = 0;
    for (auto const& e : elements)
    {
        offset.push_back(n);
        n += e.neuralNetworks.at(id).getNumConnections();
    }

    return offset;
}

void Training::dPdc(string                  property,
                    Structure&              structure,
                    vector<vector<double>>& dPdc)
{
    auto npe = getNumConnectionsPerElement();
    auto off = getConnectionOffsets();
    dPdc.clear();

    if (property == "energy")
    {
        dPdc.resize(1);
        dPdc.at(0).resize(getNumConnections(), 0.0);
        for (auto const& a : structure.atoms)
        {
            size_t e = a.element;
            NeuralNetwork& nn = elements.at(e).neuralNetworks.at(nnId);
            nn.setInput(a.G.data());
            nn.propagate();
            vector<double> tmp(npe.at(e), 0.0);
            nn.calculateDEdc(tmp.data());
            for (size_t j = 0; j < tmp.size(); ++j)
            {
                dPdc.at(0).at(off.at(e) + j) += tmp.at(j);
            }
        }
    }
    else if (property == "force")
    {
        dPdc.resize(3 * structure.numAtoms);
        size_t count = 0;
        for (size_t ia = 0; ia < structure.numAtoms; ++ia)
        {
            for (size_t ixyz = 0; ixyz < 3; ++ixyz)
            {
                dPdc.at(count).resize(getNumConnections(), 0.0);
                for (auto& a : structure.atoms)
                {
#ifndef N2P2_FULL_SFD_MEMORY
                    collectDGdxia(a, ia, ixyz);
#else
                    a.collectDGdxia(ia, ixyz);
#endif
                    size_t e = a.element;
                    NeuralNetwork& nn = elements.at(e).neuralNetworks.at(nnId);
                    nn.setInput(a.G.data());
                    nn.propagate();
                    nn.calculateDEdG(a.dEdG.data());
                    nn.getOutput(&(a.energy));
                    vector<double> tmp(npe.at(e), 0.0);
#ifndef N2P2_FULL_SFD_MEMORY
                    nn.calculateDFdc(tmp.data(), dGdxia.data());
#else
                    nn.calculateDFdc(tmp.data(), a.dGdxia.data());
#endif
                    for (size_t j = 0; j < tmp.size(); ++j)
                    {
                        dPdc.at(count).at(off.at(e) + j) += tmp.at(j);
                    }
                }
                count++;
            }
        }
    }
    else
    {
        throw runtime_error("ERROR: Weight derivatives not implemented for "
                            "property \"" + property + "\".\n");
    }

    return;
}

void Training::dPdcN(string                  property,
                     Structure&              structure,
                     vector<vector<double>>& dPdc,
                     double                  delta)
{
    auto npe = getNumConnectionsPerElement();
    auto off = getConnectionOffsets();
    dPdc.clear();

    if (property == "energy")
    {
        dPdc.resize(1);
        for (size_t ie = 0; ie < numElements; ++ie)
        {
            for (size_t ic = 0; ic < npe.at(ie); ++ic)
            {
                size_t const o = off.at(ie) + ic;
                double const w = weights.at(0).at(o);

                weights.at(0).at(o) += delta;
                setWeights();
                calculateAtomicNeuralNetworks(structure, false);
                calculateEnergy(structure);
                double energyHigh = structure.energy;

                weights.at(0).at(o) -= 2.0 * delta;
                setWeights();
                calculateAtomicNeuralNetworks(structure, false);
                calculateEnergy(structure);
                double energyLow = structure.energy;

                dPdc.at(0).push_back((energyHigh - energyLow) / (2.0 * delta));
                weights.at(0).at(o) = w;
            }
        }
    }
    else if (property == "force")
    {
        size_t count = 0;
        dPdc.resize(3 * structure.numAtoms);
        for (size_t ia = 0; ia < structure.numAtoms; ++ia)
        {
            for (size_t ixyz = 0; ixyz < 3; ++ixyz)
            {
                for (size_t ie = 0; ie < numElements; ++ie)
                {
                    for (size_t ic = 0; ic < npe.at(ie); ++ic)
                    {
                        size_t const o = off.at(ie) + ic;
                        double const w = weights.at(0).at(o);

                        weights.at(0).at(o) += delta;
                        setWeights();
                        calculateAtomicNeuralNetworks(structure, true);
                        calculateForces(structure);
                        double forceHigh = structure.atoms.at(ia).f[ixyz];

                        weights.at(0).at(o) -= 2.0 * delta;
                        setWeights();
                        calculateAtomicNeuralNetworks(structure, true);
                        calculateForces(structure);
                        double forceLow = structure.atoms.at(ia).f[ixyz];

                        dPdc.at(count).push_back((forceHigh - forceLow)
                                                 / (2.0 * delta));
                        weights.at(0).at(o) = w;
                    }
                }
                count++;
            }
        }
    }
    else
    {
        throw runtime_error("ERROR: Numeric weight derivatives not "
                            "implemented for property \""
                            + property + "\".\n");
    }

    return;
}

bool Training::advance() const
{
    if (epoch < numEpochs) return true;
    else return false;
}

void Training::getWeights()
{
    if (updateStrategy == US_COMBINED)
    {
        size_t pos = 0;
        for (size_t i = 0; i < numElements; ++i)
        {
            NeuralNetwork const& nn = elements.at(i).neuralNetworks.at(nnId);
            nn.getConnections(&(weights.at(0).at(pos)));
            pos += nn.getNumConnections();
        }
    }
    else if (updateStrategy == US_ELEMENT)
    {
        for (size_t i = 0; i < numElements; ++i)
        {
            NeuralNetwork const& nn = elements.at(i).neuralNetworks.at(nnId);
            nn.getConnections(&(weights.at(i).front()));
        }
    }

    return;
}

void Training::setWeights()
{
    if (updateStrategy == US_COMBINED)
    {
        size_t pos = 0;
        for (size_t i = 0; i < numElements; ++i)
        {
            NeuralNetwork& nn = elements.at(i).neuralNetworks.at(nnId);
            nn.setConnections(&(weights.at(0).at(pos)));
            pos += nn.getNumConnections();
        }
    }
    else if (updateStrategy == US_ELEMENT)
    {
        for (size_t i = 0; i < numElements; ++i)
        {
            NeuralNetwork& nn = elements.at(i).neuralNetworks.at(nnId);
            nn.setConnections(&(weights.at(i).front()));
        }
    }

    return;
}

// Doxygen requires namespace prefix for arguments...
void Training::addTrainingLogEntry(int                 proc,
                                   std::size_t         il,
                                   double              f,
                                   std::size_t         isg,
                                   std::size_t         is)
{
    string s = strpr("  E %5zu %10zu %5d %3zu %10.2E %10zu %5zu\n",
                     epoch, countUpdates, proc, il + 1, f, isg, is);
    trainingLog << s;

    return;
}

// Doxygen requires namespace prefix for arguments...
void Training::addTrainingLogEntry(int                 proc,
                                   std::size_t         il,
                                   double              f,
                                   std::size_t         isg,
                                   std::size_t         is,
                                   std::size_t         ia,
                                   std::size_t         ic)
{
    string s = strpr("  F %5zu %10zu %5d %3zu %10.2E %10zu %5zu %5zu %2zu\n",
                     epoch, countUpdates, proc, il + 1, f, isg, is, ia, ic);
    trainingLog << s;

    return;
}

// Doxygen requires namespace prefix for arguments...
void Training::addTrainingLogEntry(int                 proc,
                                   std::size_t         il,
                                   double              f,
                                   std::size_t         isg,
                                   std::size_t         is,
                                   std::size_t         ia)
{
    string s = strpr("  Q %5zu %10zu %5d %3zu %10.2E %10zu %5zu %5zu\n",
                     epoch, countUpdates, proc, il + 1, f, isg, is, ia);
    trainingLog << s;

    return;
}

#ifndef N2P2_FULL_SFD_MEMORY
void Training::collectDGdxia(Atom const& atom,
                             size_t      indexAtom,
                             size_t      indexComponent)
{
    size_t const nsf = atom.numSymmetryFunctions;

    // Reset dGdxia array.
    dGdxia.clear();
    vector<double>(dGdxia).swap(dGdxia);
    dGdxia.resize(nsf, 0.0);

    vector<vector<size_t> > const& tableFull
        = elements.at(atom.element).getSymmetryFunctionTable();

    for (size_t i = 0; i < atom.numNeighbors; i++)
    {
        if (atom.neighbors[i].index == indexAtom)
        {
            Atom::Neighbor const& n = atom.neighbors[i];
            vector<size_t> const& table = tableFull.at(n.element);
            for (size_t j = 0; j < n.dGdr.size(); ++j)
            {
                dGdxia[table.at(j)] += n.dGdr[j][indexComponent];
            }
        }
    }
    if (atom.index == indexAtom)
    {
        for (size_t i = 0; i < nsf; ++i)
        {
            dGdxia[i] += atom.dGdr[i][indexComponent];
        }
    }

    return;
}
#endif

void Training::randomizeNeuralNetworkWeights(string const& type)
{
    string keywordNW = "";
    if      (type == "short" ) keywordNW = "nguyen_widrow_weights_short";
    else if (type == "charge") keywordNW = "nguyen_widrow_weights_charge";
    else
    {
        throw runtime_error("ERROR: Unknown neural network type.\n");
    }

    double minWeights = atof(settings["weights_min"].c_str());
    double maxWeights = atof(settings["weights_max"].c_str());
    log << strpr("Initial weights selected randomly in interval "
                 "[%f, %f).\n", minWeights, maxWeights);
    vector<double> w;
    for (size_t i = 0; i < numElements; ++i)
    {
        NeuralNetwork& nn = elements.at(i).neuralNetworks.at(type);
        w.resize(nn.getNumConnections(), 0);
        for (size_t j = 0; j < w.size(); ++j)
        {
            w.at(j) = minWeights + gsl_rng_uniform(rngGlobal)
                    * (maxWeights - minWeights);
        }
        nn.setConnections(&(w.front()));
    }
    if (settings.keywordExists(keywordNW))
    {
        log << "Weights modified according to Nguyen Widrow scheme.\n";
        for (vector<Element>::iterator it = elements.begin();
             it != elements.end(); ++it)
        {
            NeuralNetwork& nn = it->neuralNetworks.at(type);
            nn.modifyConnections(NeuralNetwork::MS_NGUYENWIDROW);
        }
    }
    else if (settings.keywordExists("precondition_weights"))
    {
        throw runtime_error("ERROR: Preconditioning of weights not yet"
                            " implemented.\n");
    }
    else
    {
        log << "Weights modified accoring to Glorot Bengio scheme.\n";
        //log << "Weights connected to output layer node set to zero.\n";
        log << "Biases set to zero.\n";
        for (vector<Element>::iterator it = elements.begin();
             it != elements.end(); ++it)
        {
            NeuralNetwork& nn = it->neuralNetworks.at(type);
            nn.modifyConnections(NeuralNetwork::MS_GLOROTBENGIO);
            //nn->modifyConnections(NeuralNetwork::MS_ZEROOUTPUTWEIGHTS);
            nn.modifyConnections(NeuralNetwork::MS_ZEROBIAS);
        }
    }

    return;
}

void Training::setupSelectionMode(string const& property)
{
    bool all = (property == "all");
    bool isProperty = (find(pk.begin(), pk.end(), property) != pk.end());
    if (!(all || isProperty))
    {
        throw runtime_error("ERROR: Unknown property for selection mode"
                            " setup.\n");
    }

    if (all)
    {
        if (!(settings.keywordExists("selection_mode") ||
              settings.keywordExists("rmse_threshold") ||
              settings.keywordExists("rmse_threshold_trials"))) return;
        log << "Global selection mode settings:\n";
    }
    else
    {
        if (!(settings.keywordExists("selection_mode_" + property) ||
              settings.keywordExists("rmse_threshold_" + property) ||
              settings.keywordExists("rmse_threshold_trials_"
                                     + property))) return;
        log << "Selection mode settings specific to property \""
            << property << "\":\n";
    }
    string keyword;
    if (all) keyword = "selection_mode";
    else keyword = "selection_mode_" + property;

    if (settings.keywordExists(keyword))
    {
        map<size_t, SelectionMode> schedule;
        vector<string> args = split(settings[keyword]);
        if (args.size() % 2 != 1)
        {
            throw runtime_error("ERROR: Incorrect selection mode format.\n");
        }
        schedule[0] = (SelectionMode)atoi(args.at(0).c_str());
        for (size_t i = 1; i < args.size(); i = i + 2)
        {
            schedule[(size_t)atoi(args.at(i).c_str())] =
                (SelectionMode)atoi(args.at(i + 1).c_str());
        }
        for (map<size_t, SelectionMode>::const_iterator it = schedule.begin();
             it != schedule.end(); ++it)
        {
            log << strpr("- Selection mode starting with epoch %zu:\n",
                         it->first);
            if (it->second == SM_RANDOM)
            {
                log << strpr("  Random selection of update candidates: "
                             "SelectionMode::SM_RANDOM (%d)\n", it->second);
            }
            else if (it->second == SM_SORT)
            {
                log << strpr("  Update candidates selected according to error: "
                             "SelectionMode::SM_SORT (%d)\n", it->second);
            }
            else if (it->second == SM_THRESHOLD)
            {
                log << strpr("  Update candidates chosen randomly above RMSE "
                             "threshold: SelectionMode::SM_THRESHOLD (%d)\n",
                             it->second);
            }
            else
            {
                throw runtime_error("ERROR: Unknown selection mode.\n");
            }
        }
        if (all)
        {
            for (auto& i : p)
            {
                i.second.selectionModeSchedule = schedule;
                i.second.selectionMode = schedule[0];
            }
        }
        else
        {
            p[property].selectionModeSchedule = schedule;
            p[property].selectionMode = schedule[0];
        }
    }

    if (all) keyword = "rmse_threshold";
    else keyword = "rmse_threshold_" + property;
    if (settings.keywordExists(keyword))
    {
        double t = atof(settings[keyword].c_str());
        log << strpr("- RMSE selection threshold: %.2f * RMSE\n", t);
        if (all) for (auto& i : p) i.second.rmseThreshold = t;
        else p[property].rmseThreshold = t;
    }

    if (all) keyword = "rmse_threshold_trials";
    else keyword = "rmse_threshold_trials_" + property;
    if (settings.keywordExists(keyword))
    {
        size_t t = atoi(settings[keyword].c_str());
        log << strpr("- RMSE selection trials   : %zu\n", t);
        if (all) for (auto& i : p) i.second.rmseThresholdTrials = t;
        else p[property].rmseThresholdTrials = t;
    }

    return;
}

void Training::setupFileOutput(string const& type)
{
    string keyword = "write_";
    bool isProperty = (find(pk.begin(), pk.end(), type) != pk.end());
    if      (type == "energy"       ) keyword += "trainpoints";
    else if (type == "force"        ) keyword += "trainforces";
    else if (type == "charge"       ) keyword += "traincharges";
    else if (type == "weights_epoch") keyword += type;
    else if (type == "neuronstats"  ) keyword += type;
    else
    {
        throw runtime_error("ERROR: Invalid type for file output setup.\n");
    }

    // Check how often energy comparison files should be written.
    if (settings.keywordExists(keyword))
    {
        size_t* writeEvery = nullptr;
        size_t* writeAlways = nullptr;
        string message;
        if (isProperty)
        {
            writeEvery = &(p[type].writeCompEvery);
            writeAlways = &(p[type].writeCompAlways);
            message = "Property \"" + type + "\" comparison";
            message.at(0) = toupper(message.at(0));
        }
        else if (type == "weights_epoch")
        {
            writeEvery = &writeWeightsEvery;
            writeAlways = &writeWeightsAlways;
            message = "Weight";
        }
        else if (type == "neuronstats")
        {
            writeEvery = &writeNeuronStatisticsEvery;
            writeAlways = &writeNeuronStatisticsAlways;
            message = "Neuron statistics";
        }

        *writeEvery = 1;
        vector<string> v = split(reduce(settings[keyword]));
        if (v.size() == 1) *writeEvery = (size_t)atoi(v.at(0).c_str());
        else if (v.size() == 2)
        {
            *writeEvery = (size_t)atoi(v.at(0).c_str());
            *writeAlways = (size_t)atoi(v.at(1).c_str());
        }
        log << strpr((message
                      + " files will be written every %zu epochs.\n").c_str(),
                     *writeEvery);
        if (*writeAlways > 0)
        {
            log << strpr((message
                          + " files will always be written up to epoch "
                            "%zu.\n").c_str(), *writeAlways);
        }
    }

    return;
}

void Training::setupUpdatePlan(string const& property)
{
    bool isProperty = (find(pk.begin(), pk.end(), property) != pk.end());
    if (!isProperty)
    {
        throw runtime_error("ERROR: Unknown property for update plan"
                            " setup.\n");
    }

    // Actual property modified here.
    Property& pa = p[property];
    string keyword = property + "_fraction";

    // Override force fraction if keyword "energy_force_ratio" is provided.
    if (property == "force" &&
        p.exists("energy") &&
        settings.keywordExists("force_energy_ratio"))
    {
        double const ratio = atof(settings["force_energy_ratio"].c_str());
        if (settings.keywordExists(keyword))
        {
            log << "WARNING: Given force fraction is ignored because "
                   "force/energy ratio is provided.\n";
        }
        log << strpr("Desired force/energy update ratio              : %.6f\n",
                     ratio);
        log << "----------------------------------------------\n";
        pa.epochFraction = (p["energy"].numTrainPatterns * ratio)
                         / p["force"].numTrainPatterns;
    }
    // Default action = read "<property>_fraction" keyword.
    else
    {
        pa.epochFraction = atof(settings[keyword].c_str());
    }

    keyword = "task_batch_size_" + property;
    pa.taskBatchSize = (size_t)atoi(settings[keyword].c_str());
    if (pa.taskBatchSize == 0)
    {
        pa.patternsPerUpdate =
            static_cast<size_t>(pa.updateCandidates.size() * pa.epochFraction);
        pa.numUpdates = 1;
    }
    else
    {
        pa.patternsPerUpdate = pa.taskBatchSize;
        pa.numUpdates =
            static_cast<size_t>((pa.numTrainPatterns * pa.epochFraction)
                                / pa.taskBatchSize / numProcs);
    }
    pa.patternsPerUpdateGlobal = pa.patternsPerUpdate;
    MPI_Allreduce(MPI_IN_PLACE, &(pa.patternsPerUpdateGlobal), 1, MPI_SIZE_T, MPI_SUM, comm);
    pa.errorsPerTask.resize(numProcs, 0);
    if (jacobianMode == JM_FULL)
    {
        pa.errorsPerTask.at(myRank) = static_cast<int>(pa.patternsPerUpdate);
    }
    else
    {
        pa.errorsPerTask.at(myRank) = 1;
    }
    MPI_Allgather(MPI_IN_PLACE, 1, MPI_INT, &(pa.errorsPerTask.front()), 1, MPI_INT, comm);
    if (jacobianMode == JM_FULL)
    {
        pa.weightsPerTask.resize(numUpdaters);
        for (size_t i = 0; i < numUpdaters; ++i)
        {
            pa.weightsPerTask.at(i).resize(numProcs, 0);
            for (int j = 0; j < numProcs; ++j)
            {
                pa.weightsPerTask.at(i).at(j) = pa.errorsPerTask.at(j)
                                              * numWeightsPerUpdater.at(i);
            }
        }
    }
    pa.numErrorsGlobal = 0;
    for (size_t i = 0; i < pa.errorsPerTask.size(); ++i)
    {
        pa.offsetPerTask.push_back(pa.numErrorsGlobal);
        pa.numErrorsGlobal += pa.errorsPerTask.at(i);
    }
    pa.offsetJacobian.resize(numUpdaters);
    for (size_t i = 0; i < numUpdaters; ++i)
    {
        for (size_t j = 0; j < pa.offsetPerTask.size(); ++j)
        {
            pa.offsetJacobian.at(i).push_back(pa.offsetPerTask.at(j) *
                                              numWeightsPerUpdater.at(i));
        }
    }
    log << "Update plan for property \"" + property + "\":\n";
    log << strpr("- Per-task batch size                          : %zu\n",
                 pa.taskBatchSize);
    log << strpr("- Fraction of patterns used per epoch          : %.6f\n",
                 pa.epochFraction);
    if (pa.numUpdates == 0)
    {
        log << "WARNING: No updates are planned for this property.";
    }
    log << strpr("- Updates per epoch                            : %zu\n",
                 pa.numUpdates);
    log << strpr("- Patterns used per update (rank %3d / global) : "
                 "%10zu / %zu\n",
                 myRank, pa.patternsPerUpdate, pa.patternsPerUpdateGlobal);
    log << "----------------------------------------------\n";

    return;
}

void Training::allocateArrays(string const& property)
{
    bool isProperty = (find(pk.begin(), pk.end(), property) != pk.end());
    if (!isProperty)
    {
        throw runtime_error("ERROR: Unknown property for array allocation.\n");
    }

    log << "Allocating memory for " + property +
           " error vector and Jacobian.\n";
    Property& pa = p[property];
    pa.error.resize(numUpdaters);
    pa.jacobian.resize(numUpdaters);
    for (size_t i = 0; i < numUpdaters; ++i)
    {
        size_t size = 1;
        if (( parallelMode == PM_TRAIN_ALL ||
              (parallelMode == PM_TRAIN_RK0 && myRank == 0)) &&
            jacobianMode != JM_SUM)
        {
            size *= pa.numErrorsGlobal;
        }
        else if ((parallelMode == PM_TRAIN_RK0 && myRank != 0) &&
                 jacobianMode != JM_SUM)
        {
            size *= pa.errorsPerTask.at(myRank);
        }
        pa.error.at(i).resize(size, 0.0);
        pa.jacobian.at(i).resize(size * numWeightsPerUpdater.at(i), 0.0);
        log << strpr("Updater %3zu:\n", i);
        log << strpr(" - Error    size: %zu\n", pa.error.at(i).size());
        log << strpr(" - Jacobian size: %zu\n", pa.jacobian.at(i).size());
    }
    log << "----------------------------------------------\n";

    return;
}

void Training::writeTimingData(bool append, string const fileName)
{
    ofstream file;
    string fileNameActual = fileName;
    if (nnpType == NNPType::SHORT_CHARGE_NN)
    {
        fileNameActual += strpr(".stage-%zu", stage);
    }

    vector<string> sub = {"_err", "_com", "_upd", "_log"};
    if (append) file.open(fileNameActual.c_str(), ofstream::app);
    else
    {
        file.open(fileNameActual.c_str());

        // File header.
        vector<string> title;
        vector<string> colName;
        vector<string> colInfo;
        vector<size_t> colSize;
        title.push_back("Timing data for training loop.");
        colSize.push_back(10);
        colName.push_back("epoch");
        colInfo.push_back("Current epoch.");
        colSize.push_back(11);
        colName.push_back("train");
        colInfo.push_back("Time for training.");
        colSize.push_back(7);
        colName.push_back("ptrain");
        colInfo.push_back("Time for training (percentage of loop).");
        colSize.push_back(11);
        colName.push_back("error");
        colInfo.push_back("Time for error calculation.");
        colSize.push_back(7);
        colName.push_back("perror");
        colInfo.push_back("Time for error calculation (percentage of loop).");
        colSize.push_back(11);
        colName.push_back("epoch");
        colInfo.push_back("Time for this epoch.");
        colSize.push_back(11);
        colName.push_back("total");
        colInfo.push_back("Total time for all epochs.");
        for (auto k : pk)
        {
            colSize.push_back(11);
            colName.push_back(p[k].tiny + "train");
            colInfo.push_back("");
            colSize.push_back(7);
            colName.push_back(p[k].tiny + "ptrain");
            colInfo.push_back("");
        }
        for (auto s : sub)
        {
            for (auto k : pk)
            {
                colSize.push_back(11);
                colName.push_back(p[k].tiny + s);
                colInfo.push_back("");
                colSize.push_back(7);
                colName.push_back(p[k].tiny + "p" + s);
                colInfo.push_back("");
            }
        }
        appendLinesToFile(file,
                          createFileHeader(title, colSize, colName, colInfo));
    }

    double timeLoop = sw["loop"].getLoop();
    file << strpr("%10zu", epoch);
    file << strpr(" %11.3E", sw["train"].getLoop());
    file << strpr(" %7.3f", sw["train"].getLoop() / timeLoop);
    file << strpr(" %11.3E", sw["error"].getLoop());
    file << strpr(" %7.3f", sw["error"].getLoop() / timeLoop);
    file << strpr(" %11.3E", timeLoop);
    file << strpr(" %11.3E", sw["loop"].getTotal());

    for (auto k : pk)
    {
        file << strpr(" %11.3E", sw[k].getLoop());
        file << strpr(" %7.3f", sw[k].getLoop() / sw["train"].getLoop());
    }
    for (auto s : sub)
    {
        for (auto k : pk)
        {
            file << strpr(" %11.3E", sw[k + s].getLoop());
            file << strpr(" %7.3f", sw[k + s].getLoop() / sw[k].getLoop());
        }
    }
    file << "\n";

    file.flush();
    file.close();

    return;
}

Training::Property::Property(string const& property) :
    property               (property ),
    displayMetric          (""       ),
    tiny                   (""       ),
    plural                 (""       ),
    selectionMode          (SM_RANDOM),
    numTrainPatterns       (0        ),
    numTestPatterns        (0        ),
    taskBatchSize          (0        ),
    writeCompEvery         (0        ),
    writeCompAlways        (0        ),
    posUpdateCandidates    (0        ),
    rmseThresholdTrials    (0        ),
    countUpdates           (0        ),
    numUpdates             (0        ),
    patternsPerUpdate      (0        ),
    patternsPerUpdateGlobal(0        ),
    numErrorsGlobal        (0        ),
    epochFraction          (0.0      ),
    rmseThreshold          (0.0      )
{
    if (property == "energy")
    {
        tiny = "E";
        plural = "energies";
        errorMetrics = {"RMSEpa", "RMSE", "MAEpa", "MAE"};
    }
    else if (property == "force")
    {
        tiny = "F";
        plural = "forces";
        errorMetrics = {"RMSE", "MAE"};
    }
    else if (property == "charge")
    {
        tiny = "Q";
        plural = "charges";
        errorMetrics = {"RMSE", "MAE"};
    }
    else
    {
        throw runtime_error("ERROR: Unknown training property.\n");
    }

    // Set up error metrics
    for (auto m : errorMetrics)
    {
        errorTrain[m] = 0.0;
        errorTest[m] = 0.0;
    }
    displayMetric = errorMetrics.at(0);
