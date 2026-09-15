/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           |
     \\/     M anipulation  |
-------------------------------------------------------------------------------
\*---------------------------------------------------------------------------*/

#include "radexBase.H"
#include "volFields.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{
namespace functionObjects
{

defineTypeNameAndDebug(radexBase, 0);

} // End namespace functionObjects
} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

Foam::functionObjects::radexBase::radexBase
(
    const word& name,
    const Time& runTime,
    const dictionary& dict
)
:
    fvMeshFunctionObject(name, runTime, dict),
    fieldNames_(),
    scalarNames_(),
    io_
    (
        dict.getOrDefault<word>("backend", "dragon"),
        dict.getOrDefault<word>("identifier", "")
    )
{}


Foam::functionObjects::radexBase::~radexBase() = default;


bool Foam::functionObjects::radexBase::read(const dictionary& dict)
{
    if (!fvMeshFunctionObject::read(dict))
    {
        return false;
    }

    dict.readEntry("fields", fieldNames_);
    dict.readIfPresent("scalars", scalarNames_);
    radexIO::setIdentifier(dict.getOrDefault<word>("identifier", ""));

    Info<< type() << " " << name() << ": fields = " << fieldNames_
        << ", scalars = " << scalarNames_
        << ", identifier = " << radexIO::identifier() << nl;

    return true;
}


Foam::label Foam::functionObjects::radexBase::subdomainId()
{
    return radexIO::subdomainId();
}


std::string Foam::functionObjects::radexBase::makeKey
(
    const word& fieldName,
    label subdomainId
) const
{
    const std::string key =
        io_.makeKey(fieldName, mesh_.time().timeName(), subdomainId);

    Info<< type() << " " << actionName() << " " << key << nl;

    return key;
}


bool Foam::functionObjects::radexBase::processFields()
{
    const label subdomain = subdomainId();

    for (const word& fieldName : fieldNames_)
    {
        if (mesh_.foundObject<volScalarField>(fieldName))
        {
            processScalarField(fieldName, subdomain);
        }
        else if (mesh_.foundObject<volVectorField>(fieldName))
        {
            processVectorField(fieldName, subdomain);
        }
        else
        {
            WarningInFunction
                << "Field " << fieldName
                << " not found in objectRegistry; skipping" << nl;
        }
    }

    for (const word& fieldName : scalarNames_)
    {
        processUniformScalar(fieldName, subdomain);
    }

    return true;
}


// ************************************************************************* //
