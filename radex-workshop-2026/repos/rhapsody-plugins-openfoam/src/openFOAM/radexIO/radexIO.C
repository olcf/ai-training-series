/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           |
     \\/     M anipulation  |
-------------------------------------------------------------------------------
\*---------------------------------------------------------------------------*/

#include "radexIO.H"
#include "IOdictionary.H"
#include "Pstream.H"

#include "radex/client_base.hpp"
#include "radex/dragon.hpp"
#include "radex/exceptions.hpp"
#include "radex/smartredis.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <memory>

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{
namespace
{

struct SharedState
{
    //- Non-owning, so that the connection is closed with the last radexIO
    //- instance rather than during static destruction, by which time the
    //- backend may already have been finalised
    std::weak_ptr<radex::IClient> client;

    word backend;
    word identifier;
    bool identifierSet = false;
};


SharedState& sharedState()
{
    static SharedState state;
    return state;
}


std::chrono::milliseconds toMilliseconds(scalar seconds)
{
    return std::chrono::milliseconds
    (
        static_cast<std::int64_t>(seconds * 1000)
    );
}


// Deduplicated so that a mismatched producer does not warn on every transfer
bool conversionWarningIssued = false;
bool singletonWarningIssued = false;


template<class T>
void copyTensor
(
    const radex::TensorInfo<T>& info,
    List<scalar>& data,
    List<label>& shape
)
{
    data.resize(static_cast<label>(info.data.size()));
    std::copy(info.data.begin(), info.data.end(), data.begin());

    shape.resize(static_cast<label>(info.dims.size()));
    std::copy(info.dims.begin(), info.dims.end(), shape.begin());
}

} // End anonymous namespace
} // End namespace Foam


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

std::shared_ptr<radex::IClient>
Foam::radexIO::acquireClient(const word& backend)
{
    word effectiveBackend(backend);

    if (backend != "redis" && backend != "dragon")
    {
        WarningInFunction
            << "Unknown radex backend '" << backend
            << "'; defaulting to dragon" << nl;

        effectiveBackend = "dragon";
    }

    SharedState& state = sharedState();

    if (auto client = state.client.lock())
    {
        if (effectiveBackend != state.backend)
        {
            WarningInFunction
                << "radex backend '" << effectiveBackend
                << "' requested, but this process is already connected via '"
                << state.backend
                << "'. All radex users share one connection;"
                << " reusing the existing one" << nl;
        }

        return client;
    }

    std::shared_ptr<radex::IClient> client;

    if (effectiveBackend == "redis")
    {
        client = std::make_shared<radex::redis::smartredis::Client>();
    }
    else
    {
        client = std::make_shared<radex::drg::ddict::Client>();
    }

    state.client = client;
    state.backend = effectiveBackend;

    Info<< "radexIO: connected to the radex store via " << effectiveBackend
        << " on subdomain " << subdomainId() << nl;

    return client;
}


Foam::radexIO::radexIO(const word& backend, const word& identifier)
:
    client_(acquireClient(backend))
{
    setIdentifier(identifier);
}


Foam::radexIO::radexIO(const Time& runTime)
:
    client_(nullptr)
{
    IOdictionary dict
    (
        IOobject
        (
            "radexDict",
            runTime.system(),
            runTime,
            IOobject::READ_IF_PRESENT,
            IOobject::NO_WRITE,
            IOobject::NO_REGISTER
        )
    );

    client_ = acquireClient(dict.getOrDefault<word>("backend", "dragon"));

    setIdentifier(dict.getOrDefault<word>("identifier", ""));
}


// Out of line so that radex::IClient need only be complete here
Foam::radexIO::~radexIO() = default;


radex::IClient& Foam::radexIO::client() const
{
    return *client_;
}


const Foam::word& Foam::radexIO::backend()
{
    return sharedState().backend;
}


const Foam::word& Foam::radexIO::identifier()
{
    return sharedState().identifier;
}


void Foam::radexIO::setIdentifier(const word& identifier)
{
    if (identifier.empty())
    {
        return;
    }

    SharedState& state = sharedState();

    if (state.identifierSet)
    {
        if (identifier != state.identifier)
        {
            WarningInFunction
                << "radex identifier '" << identifier
                << "' requested, but this process is already using '"
                << state.identifier
                << "'. All radex users share one identifier;"
                << " keeping the established one" << nl;
        }

        return;
    }

    state.identifier = identifier;
    state.identifierSet = true;
}


Foam::label Foam::radexIO::subdomainId()
{
    return Pstream::myProcNo();
}


std::string Foam::radexIO::makeKey
(
    const word& name,
    const std::string& timeName,
    label subdomainId
) const
{
    return radexKey(name).withTime(timeName).withSubdomain(subdomainId).str();
}


Foam::radexKey::radexKey(const word& name)
{
    const word& id = radexIO::identifier();

    if (!id.empty())
    {
        key_ = std::string(id) + "_";
    }

    key_ += std::string(name);
}


Foam::radexKey& Foam::radexKey::withTime(const std::string& timeName)
{
    key_ += "_" + timeName;
    return *this;
}


Foam::radexKey& Foam::radexKey::withSubdomain(label subdomainId)
{
    key_ += "_" + std::to_string(subdomainId);
    return *this;
}


Foam::radexKey& Foam::radexKey::withIndex(label index)
{
    key_ += "_" + std::to_string(index);
    return *this;
}


radex::data::OutgoingHandle Foam::radexKey::outgoing() const
{
    return radex::data::OutgoingHandle(key_);
}


radex::data::IncomingHandle Foam::radexKey::incoming() const
{
    return radex::data::IncomingHandle(key_);
}


void Foam::radexIO::putTensor
(
    const radex::data::OutgoingHandle& handle,
    const scalar* data,
    label nElements,
    label nComponents
) const
{
    const radex::detail::MetaInt dims[] =
    {
        static_cast<radex::detail::MetaInt>(nElements),
        static_cast<radex::detail::MetaInt>(nComponents)
    };

    client_->put_tensor<scalar>
    (
        handle,
        dims,
        (nComponents == 1 ? 1 : 2),
        data,
        static_cast<radex::detail::MetaInt>(nElements * nComponents)
    );
}


bool Foam::radexIO::fetchTensor
(
    const radex::data::IncomingHandle& handle,
    bool wait,
    scalar timeout,
    List<scalar>& data,
    List<label>& shape
) const
{
    // contains() rather than relying on get_tensor: the non-blocking path
    // must not raise for a key that simply has not arrived yet
    if (!wait && !client_->contains(handle.key()))
    {
        WarningInFunction
            << "Key " << handle.key()
            << " is not in the store; skipping" << nl;

        return false;
    }

    try
    {
        copyTensor
        (
            wait
          ? client_->wait_for_tensor<scalar>(handle, toMilliseconds(timeout))
          : client_->get_tensor<scalar>(handle),
            data,
            shape
        );
    }
    catch (const radex::DTypeMismatchError&)
    {
        copyTensor
        (
            wait
          ? client_->wait_for_tensor<otherScalar>
            (
                handle, toMilliseconds(timeout)
            )
          : client_->get_tensor<otherScalar>(handle),
            data,
            shape
        );

        if (!conversionWarningIssued)
        {
            WarningInFunction
                << "Tensor " << handle.key() << " is stored as "
                << sizeof(otherScalar) * 8 << "-bit floating point but this"
                << " solver uses " << sizeof(scalar) * 8 << "-bit; converting."
                << " Further occurrences will not be reported." << nl;

            conversionWarningIssued = true;
        }
    }

    return true;
}


bool Foam::radexIO::shapeMatches
(
    const List<label>& shape,
    label nElements,
    label nComponents,
    const std::string& key
) const
{
    List<label> effective(shape);

    while (effective.size() > 1 && effective.last() == 1)
    {
        effective.resize(effective.size() - 1);
    }

    if (effective.size() != shape.size() && !singletonWarningIssued)
    {
        WarningInFunction
            << "Tensor " << key << " has shape " << shape
            << "; treating it as " << effective
            << ". Further occurrences will not be reported." << nl;

        singletonWarningIssued = true;
    }

    const label expectedRank = (nComponents == 1 ? 1 : 2);

    const bool ok =
        effective.size() == expectedRank
     && effective[0] == nElements
     && (expectedRank == 1 || effective[1] == nComponents);

    if (!ok)
    {
        WarningInFunction
            << "Tensor " << key << " has shape " << shape
            << "; expected ";

        if (expectedRank == 1)
        {
            WarningInFunction << "[" << nElements << "]";
        }
        else
        {
            WarningInFunction << "[" << nElements << ", " << nComponents << "]";
        }

        WarningInFunction
            << ". A differing leading extent usually means the case was"
            << " decomposed differently when the data was written."
            << " Skipping" << nl;
    }

    return ok;
}


void Foam::radexIO::put
(
    const radex::data::OutgoingHandle& handle,
    scalar value
) const
{
    client_->put_scalar<scalar>(handle, value);
}


bool Foam::radexIO::get
(
    const radex::data::IncomingHandle& handle,
    scalar& value,
    bool wait,
    scalar timeout
) const
{
    if (!wait && !client_->contains(handle.key()))
    {
        WarningInFunction
            << "Key " << handle.key()
            << " is not in the store; skipping" << nl;

        return false;
    }

    try
    {
        value = wait
          ? client_->wait_for_scalar<scalar>(handle, toMilliseconds(timeout))
          : client_->get_scalar<scalar>(handle);
    }
    catch (const radex::DTypeMismatchError&)
    {
        value = static_cast<scalar>
        (
            wait
          ? client_->wait_for_scalar<otherScalar>
            (
                handle, toMilliseconds(timeout)
            )
          : client_->get_scalar<otherScalar>(handle)
        );

        if (!conversionWarningIssued)
        {
            WarningInFunction
                << "Scalar " << handle.key() << " is stored as "
                << sizeof(otherScalar) * 8 << "-bit floating point but this"
                << " solver uses " << sizeof(scalar) * 8 << "-bit; converting."
                << " Further occurrences will not be reported." << nl;

            conversionWarningIssued = true;
        }
    }

    return true;
}


// ************************************************************************* //
