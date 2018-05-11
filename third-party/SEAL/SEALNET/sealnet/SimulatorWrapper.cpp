#include "sealnet/SimulatorWrapper.h"
#include "sealnet/EncryptionParamsWrapper.h"
#include "sealnet/BigUIntWrapper.h"
#include "sealnet/Common.h"
#include <msclr\marshal.h>
#include <msclr\marshal_cppstd.h>

using namespace System;
using namespace System::Collections::Generic;
using namespace std;
using namespace msclr::interop;

namespace Microsoft
{
    namespace Research
    {
        namespace SEAL
        {
            int Simulation::InvariantNoiseBudget::get()
            {
                if (simulation_ == nullptr)
                {
                    throw gcnew ObjectDisposedException("Simulation is disposed");
                }
                return simulation_->invariant_noise_budget();
            }

            int Simulation::Size::get()
            {
                if (simulation_ == nullptr)
                {
                    throw gcnew ObjectDisposedException("Simulation is disposed");
                }
                return simulation_->size();
            }

            bool Simulation::Decrypts()
            {
                if (simulation_ == nullptr)
                {
                    throw gcnew ObjectDisposedException("Simulation is disposed");
                }
                try
                {
                    return simulation_->decrypts();
                }
                catch (const exception &e)
                {
                    HandleException(&e);
                }
                catch (...)
                {
                    HandleException(nullptr);
                }
                throw gcnew Exception("Unexpected exception");
            }

            bool Simulation::Decrypts(int budgetGap)
            {
                if (simulation_ == nullptr)
                {
                    throw gcnew ObjectDisposedException("Simulation is disposed");
                }
                try
                {
                    return simulation_->decrypts(budgetGap);
                }
                catch (const exception &e)
                {
                    HandleException(&e);
                }
                catch (...)
                {
                    HandleException(nullptr);
                }
                throw gcnew Exception("Unexpected exception");
            }

            Simulation::Simulation(EncryptionParameters ^parms, int ciphertextSize, int noiseBudget)
            {
                if (parms == nullptr)
                {
                    throw gcnew ArgumentNullException("parms cannot be null");
                }
                try
                {
                    simulation_ = new seal::Simulation(parms->GetParms(), ciphertextSize, noiseBudget);
                    GC::KeepAlive(parms);
                }
                catch (const exception &e)
                {
                    HandleException(&e);
                }
                catch (...)
                {
                    HandleException(nullptr);
                }
            }

            Simulation::Simulation(const seal::Simulation &copy)
            {
                try
                {
                    simulation_ = new seal::Simulation(copy);
                }
                catch (const exception &e)
                {
                    HandleException(&e);
                }
                catch (...)
                {
                    HandleException(nullptr);
                }
            }

            seal::Simulation &Simulation::GetSimulation()
            {
                if (simulation_ == nullptr)
                {
                    throw gcnew ObjectDisposedException("Simulation is disposed");
                }
                return *simulation_;
            }

            Simulation::~Simulation()
            {
                this->!Simulation();
            }

            Simulation::!Simulation()
            {
                if (simulation_ != nullptr)
                {
                    delete simulation_;
                    simulation_ = nullptr;
                }
            }

            SimulationEvaluator::SimulationEvaluator()
            {
                simulationEvaluator_ = new seal::SimulationEvaluator();
            }

            SimulationEvaluator::SimulationEvaluator(MemoryPoolHandle ^pool)
            {
                if (pool == nullptr)
                {
                    throw gcnew ArgumentNullException("pool cannot be null");
                }
                simulationEvaluator_ = new seal::SimulationEvaluator(pool->GetHandle());
            }

            SimulationEvaluator::~SimulationEvaluator()
            {
                this->!SimulationEvaluator();
            }

            SimulationEvaluator::!SimulationEvaluator()
            {
                if (simulationEvaluator_ != nullptr)
                {
                    delete simulationEvaluator_;
                    simulationEvaluator_ = nullptr;
                }
            }

            Simulation ^SimulationEvaluator::GetFresh(EncryptionParameters ^parms, int plainMaxCoeffCount, UInt64 plainMaxAbsValue)
            {
                if (simulationEvaluator_ == nullptr)
                {
                    throw gcnew ObjectDisposedException("SimulationEvaluator is disposed");
                }
                if (parms == nullptr)
                {
                    throw gcnew ArgumentNullException("parms cannot be null");
                }
                try
                {
                    auto result = gcnew Simulation(simulationEvaluator_->get_fresh(parms->GetParms(), 
                        plainMaxCoeffCount, plainMaxAbsValue));
                    GC::KeepAlive(parms);
                    return result;
                }
                catch (const exception &e)
                {
                    HandleException(&e);
                }
                catch (...)
                {
                    HandleException(nullptr);
                }
                throw gcnew Exception("Unexpected exception");
            }

            Simulation ^SimulationEvaluator::Negate(Simulation ^simulation)
            {
                if (simulationEvaluator_ == nullptr)
                {
                    throw gcnew ObjectDisposedException("SimulationEvaluator is disposed");
                }
                if (simulation == nullptr)
                {
                    throw gcnew ArgumentNullException("simulation cannot be null");
                }
                try
                {
                    auto result = gcnew Simulation(simulationEvaluator_->negate(simulation->GetSimulation()));
                    GC::KeepAlive(simulation);
                    return result;
                }
                catch (const exception &e)
                {
                    HandleException(&e);
                }
                catch (...)
                {
                    HandleException(nullptr);
                }
                throw gcnew Exception("Unexpected exception");
            }

            Simulation ^SimulationEvaluator::Add(Simulation ^simulation1, Simulation ^simulation2)
            {
                if (simulationEvaluator_ == nullptr)
                {
                    throw gcnew ObjectDisposedException("SimulationEvaluator is disposed");
                }
                if (simulation1 == nullptr)
                {
                    throw gcnew ArgumentNullException("simulation1 cannot be null");
                }
                if (simulation2 == nullptr)
                {
                    throw gcnew ArgumentNullException("simulation2 cannot be null");
                }
                try
                {
                    auto result = gcnew Simulation(simulationEvaluator_->add(simulation1->GetSimulation(), 
                        simulation2->GetSimulation()));
                    GC::KeepAlive(simulation1);
                    GC::KeepAlive(simulation2);
                    return result;
                }
                catch (const exception &e)
                {
                    HandleException(&e);
                }
                catch (...)
                {
                    HandleException(nullptr);
                }
                throw gcnew Exception("Unexpected exception");
            }

            Simulation ^SimulationEvaluator::AddMany(List<Simulation^> ^simulations)
            {
                if (simulationEvaluator_ == nullptr)
                {
                    throw gcnew ObjectDisposedException("SimulationEvaluator is disposed");
                }
                if (simulations == nullptr)
                {
                    throw gcnew ArgumentNullException("simulations cannot be null");
                }
                try
                {
                    vector<seal::Simulation> v_simulations;
                    for each (Simulation ^simulation in simulations)
                    {
                        if (simulation == nullptr)
                        {
                            throw gcnew ArgumentNullException("simulations cannot be null");
                        }
                        v_simulations.emplace_back(simulation->GetSimulation());
                        GC::KeepAlive(simulation);
                    }
                    return gcnew Simulation(simulationEvaluator_->add_many(v_simulations));
                }
                catch (const exception& e)
                {
                    HandleException(&e);
                }
                catch (...)
                {
                    HandleException(nullptr);
                }
                throw gcnew Exception("Unexpected exception");
            }

            Simulation ^SimulationEvaluator::Sub(Simulation ^simulation1, Simulation ^simulation2)
            {
                if (simulationEvaluator_ == nullptr)
                {
                    throw gcnew ObjectDisposedException("SimulationEvaluator is disposed");
                }
                if (simulation1 == nullptr)
                {
                    throw gcnew ArgumentNullException("simulation1 cannot be null");
                }
                if (simulation2 == nullptr)
                {
                    throw gcnew ArgumentNullException("simulation2 cannot be null");
                }
                try
                {
                    auto result = gcnew Simulation(simulationEvaluator_->sub(simulation1->GetSimulation(), 
                        simulation2->GetSimulation()));
                    GC::KeepAlive(simulation1);
                    GC::KeepAlive(simulation2);
                    return result;
                }
                catch (const exception &e)
                {
                    HandleException(&e);
                }
                catch (...)
                {
                    HandleException(nullptr);
                }
                throw gcnew Exception("Unexpected exception");
            }

            Simulation ^SimulationEvaluator::AddPlain(Simulation ^simulation, int plainMaxCoeffCount, UInt64 plainMaxAbsValue)
            {
                if (simulationEvaluator_ == nullptr)
                {
                    throw gcnew ObjectDisposedException("SimulationEvaluator is disposed");
                }
                if (simulation == nullptr)
                {
                    throw gcnew ArgumentNullException("simulation cannot be null");
                }
                try
                {
                    auto result = gcnew Simulation(simulationEvaluator_->add_plain(simulation->GetSimulation(), 
                        plainMaxCoeffCount, plainMaxAbsValue));
                    GC::KeepAlive(simulation);
                    return result;
                }
                catch (const exception &e)
                {
                    HandleException(&e);
                }
                catch (...)
                {
                    HandleException(nullptr);
                }
                throw gcnew Exception("Unexpected exception");
            }

            Simulation ^SimulationEvaluator::SubPlain(Simulation ^simulation, int plainMaxCoeffCount, UInt64 plainMaxAbsValue)
            {
                if (simulationEvaluator_ == nullptr)
                {
                    throw gcnew ObjectDisposedException("SimulationEvaluator is disposed");
                }
                if (simulation == nullptr)
                {
                    throw gcnew ArgumentNullException("simulation cannot be null");
                }
                try
                {
                    auto result = gcnew Simulation(simulationEvaluator_->sub_plain(simulation->GetSimulation(), 
                        plainMaxCoeffCount, plainMaxAbsValue));
                    GC::KeepAlive(simulation);
                    return result;
                }
                catch (const exception &e)
                {
                    HandleException(&e);
                }
                catch (...)
                {
                    HandleException(nullptr);
                }
                throw gcnew Exception("Unexpected exception");
            }

            Simulation ^SimulationEvaluator::MultiplyPlain(Simulation ^simulation, int plainMaxCoeffCount, UInt64 plainMaxAbsValue)
            {
                if (simulationEvaluator_ == nullptr)
                {
                    throw gcnew ObjectDisposedException("SimulationEvaluator is disposed");
                }
                if (simulation == nullptr)
                {
                    throw gcnew ArgumentNullException("simulation cannot be null");
                }
                try
                {
                    auto result = gcnew Simulation(simulationEvaluator_->multiply_plain(simulation->GetSimulation(), 
                        plainMaxCoeffCount, plainMaxAbsValue));
                    GC::KeepAlive(simulation);
                    return result;
                }
                catch (const exception &e)
                {
                    HandleException(&e);
                }
                catch (...)
                {
                    HandleException(nullptr);
                }
                throw gcnew Exception("Unexpected exception");
            }

            Simulation ^SimulationEvaluator::Multiply(Simulation ^simulation1, Simulation ^simulation2)
            {
                if (simulationEvaluator_ == nullptr)
                {
                    throw gcnew ObjectDisposedException("SimulationEvaluator is disposed");
                }
                if (simulation1 == nullptr)
                {
                    throw gcnew ArgumentNullException("simulation1 cannot be null");
                }
                if (simulation2 == nullptr)
                {
                    throw gcnew ArgumentNullException("simulation2 cannot be null");
                }
                try
                {
                    auto result = gcnew Simulation(simulationEvaluator_->multiply(simulation1->GetSimulation(),
                        simulation2->GetSimulation()));
                    GC::KeepAlive(simulation1);
                    GC::KeepAlive(simulation2);
                    return result;
                }
                catch (const exception &e)
                {
                    HandleException(&e);
                }
                catch (...)
                {
                    HandleException(nullptr);
                }
                throw gcnew Exception("Unexpected exception");
            }

            Simulation ^SimulationEvaluator::Square(Simulation ^simulation)
            {
                if (simulationEvaluator_ == nullptr)
                {
                    throw gcnew ObjectDisposedException("SimulationEvaluator is disposed");
                }
                if (simulation == nullptr)
                {
                    throw gcnew ArgumentNullException("simulation cannot be null");
                }
                try
                {
                    auto result = gcnew Simulation(simulationEvaluator_->square(simulation->GetSimulation()));
                    GC::KeepAlive(simulation);
                    return result;
                }
                catch (const exception &e)
                {
                    HandleException(&e);
                }
                catch (...)
                {
                    HandleException(nullptr);
                }
                throw gcnew Exception("Unexpected exception");
            }

            Simulation ^SimulationEvaluator::Exponentiate(Simulation ^simulation, UInt64 exponent, int decompositionBitCount)
            {
                if (simulationEvaluator_ == nullptr)
                {
                    throw gcnew ObjectDisposedException("SimulationEvaluator is disposed");
                }
                if (simulation == nullptr)
                {
                    throw gcnew ArgumentNullException("simulation cannot be null");
                }
                try
                {
                    auto result = gcnew Simulation(simulationEvaluator_->exponentiate(simulation->GetSimulation(),
                        exponent, decompositionBitCount));
                    GC::KeepAlive(simulation);
                    return result;
                }
                catch (const exception &e)
                {
                    HandleException(&e);
                }
                catch (...)
                {
                    HandleException(nullptr);
                }
                throw gcnew Exception("Unexpected exception");
            }

            Simulation ^SimulationEvaluator::MultiplyMany(List<Simulation^> ^simulations, int decompositionBitCount)
            {
                if (simulationEvaluator_ == nullptr)
                {
                    throw gcnew ObjectDisposedException("SimulationEvaluator is disposed");
                }
                if (simulations == nullptr)
                {
                    throw gcnew ArgumentNullException("simulations cannot be null");
                }
                try
                {
                    vector<seal::Simulation> v_simulations;
                    for each (Simulation ^simulation in simulations)
                    {
                        if (simulation == nullptr)
                        {
                            throw gcnew ArgumentNullException("simulations cannot be null");
                        }
                        v_simulations.emplace_back(simulation->GetSimulation());
                        GC::KeepAlive(simulation);
                    }
                    return gcnew Simulation(simulationEvaluator_->multiply_many(v_simulations, decompositionBitCount));
                }
                catch (const exception& e)
                {
                    HandleException(&e);
                }
                catch (...)
                {
                    HandleException(nullptr);
                }
                throw gcnew Exception("Unexpected exception");
            }

            Simulation ^SimulationEvaluator::Relinearize(Simulation ^simulation, int decompositionBitCount)
            {
                if (simulationEvaluator_ == nullptr)
                {
                    throw gcnew ObjectDisposedException("SimulationEvaluator is disposed");
                }
                if (simulation == nullptr)
                {
                    throw gcnew ArgumentNullException("simulation cannot be null");
                }
                try
                {
                    auto result = gcnew Simulation(simulationEvaluator_->relinearize(simulation->GetSimulation(), 
                        decompositionBitCount));
                    GC::KeepAlive(simulation);
                    return result;
                }
                catch (const exception &e)
                {
                    HandleException(&e);
                }
                catch (...)
                {
                    HandleException(nullptr);
                }
                throw gcnew Exception("Unexpected exception");
            }
        }
    }
}