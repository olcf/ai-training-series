/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           |
     \\/     M anipulation  |
-------------------------------------------------------------------------------
\*---------------------------------------------------------------------------*/

#include "radexRead.H"
#include "addToRunTimeSelectionTable.H"
#include "volFields.H"
#include "uniformDimensionedFields.H"
#include "Pstream.H"
#include "profilingTrigger.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{
namespace functionObjects
{

defineTypeNameAndDebug(radexRead, 0);
addToRunTimeSelectionTable(functionObject, radexRead, dictionary);

} // End namespace functionObjects
} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

Foam::functionObjects::radexRead::radexRead
(
    const word& name,
    const Time& runTime,
    const dictionary& dict
)
:
    radexBase(name, runTime, dict),
    wait_(true),
    timeout_(30)
{
    read(dict);
}


bool Foam::functionObjects::radexRead::read(const dictionary& dict)
{
    if (!radexBase::read(dict))
    {
        return false;
    }

    wait_ = dict.getOrDefault<bool>("wait", true);
    timeout_ = dict.getOrDefault<scalar>("timeout", 30);

    Info<< type() << " " << name()
        << ": wait = " << wait_ << ", timeout = " << timeout_ << nl;

    return true;
}


bool Foam::functionObjects::radexRead::execute()
{
    return processFields();
}


bool Foam::functionObjects::radexRead::write()
{
    return true;
}


radex::data::IncomingHandle Foam::functionObjects::radexRead::makeHandle
(
    const word& fieldName,
    label subdomainId
) const
{
    return radex::data::IncomingHandle(makeKey(fieldName, subdomainId));
}


bool Foam::functionObjects::radexRead::processScalarField
(
    const word& fieldName,
    label subdomainId
)
{
    addProfiling(radexReadScalarField, "radexRead::readScalarField(", fieldName, ")");

    const radex::data::IncomingHandle handle = makeHandle(fieldName, subdomainId);

    return io_.get
    (
        handle,
        mesh_.lookupObjectRef<volScalarField>(fieldName),
        wait_,
        timeout_
    );
}


bool Foam::functionObjects::radexRead::processVectorField
(
    const word& fieldName,
    label subdomainId
)
{
    addProfiling(radexReadVectorField, "radexRead::readVectorField(", fieldName, ")");

    const radex::data::IncomingHandle handle = makeHandle(fieldName, subdomainId);

    return io_.get
    (
        handle,
        mesh_.lookupObjectRef<volVectorField>(fieldName),
        wait_,
        timeout_
    );
}


bool Foam::functionObjects::radexRead::processUniformScalar
(
    const word& fieldName,
    label subdomainId
)
{
    addProfiling(radexReadUniformScalar, "radexRead::readUniformScalar(", fieldName, ")");

    const radex::data::IncomingHandle handle = makeHandle(fieldName, subdomainId);

    if (!mesh_.foundObject<uniformDimensionedScalarField>(fieldName))
    {
        WarningInFunction
            << "Scalar " << fieldName
            << " not found as a registered field; skipping" << nl;
        return false;
    }

    auto& f = mesh_.lookupObjectRef<uniformDimensionedScalarField>(fieldName);

    addProfiling(radexReadUniformScalarGet, "radexRead::get_scalar(", fieldName, ")");

    return io_.get(handle, f.value(), wait_, timeout_);
}


// ************************************************************************* //
