/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           |
     \\/     M anipulation  |
-------------------------------------------------------------------------------
Description
    Reads OpenFOAM fields from a radex key-value store at every time step.

        The key naming convention is:
            <fieldName>_<timeStep>_<subdomainId>
        or (when identifier is configured):
            <identifier>_<fieldName>_<timeStep>_<subdomainId>

    Supported field types:
      - volScalarField             <- tensor of shape [nCells]  (via `fields`)
      - volVectorField             <- tensor of shape [nCells, 3] (via `fields`)
      - uniformDimensionedScalarField <- single scalar (via `scalars`)

    By default, each read blocks until the corresponding value appears in the
    store (`wait_for_*` semantics). Set `wait false;` to instead fail with a
    warning if the value is not yet present (`get_*` semantics).

    Backend is selected at runtime via the `backend` dictionary entry:
      backend   dragon;  (default)
      backend   redis;

    Example controlDict entry:
    \verbatim
    radexFields
    {
        type            radexRead;
        libs            (radexRead);

        fields          (U p nut nuTilda);
        scalars         (someAverage);

        // Optional; defaults shown
        wait            true;
        timeout         30;
    }
    \endverbatim

\*---------------------------------------------------------------------------*/

#ifndef radexRead_H
#define radexRead_H

#include "radexBase.H"

#include "radex/handles.hpp"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{
namespace functionObjects
{

/*---------------------------------------------------------------------------*\
                         Class radexRead Declaration
\*---------------------------------------------------------------------------*/

class radexRead
:
    public radexBase
{
    // Private Data

        //- If true, block on each read until the value appears
        //- (wait_for_* semantics); otherwise fetch once and warn if absent
        //- (get_* semantics)
        bool wait_;

        //- Timeout (seconds) used when wait_ is true
        scalar timeout_;


    // Private Member Functions

        //- Verb used in log messages
        virtual const char* actionName() const { return "Reading"; }

        //- Read a volScalarField from the key-value store
        virtual bool processScalarField
        (
            const word& fieldName,
            label subdomainId
        );

        //- Read a volVectorField from the key-value store
        virtual bool processVectorField
        (
            const word& fieldName,
            label subdomainId
        );

        //- Read a uniformDimensionedScalarField from the key-value store
        virtual bool processUniformScalar
        (
            const word& fieldName,
            label subdomainId
        );

        //- Build an IncomingHandle from the shared key convention
        radex::data::IncomingHandle
        makeHandle(const word& fieldName, label subdomainId) const;

        //- No copy construct
        radexRead(const radexRead&) = delete;

        //- No copy assignment
        void operator=(const radexRead&) = delete;


public:

    //- Runtime type information
    TypeName("radexRead");


    // Constructors

        //- Construct from Time and dictionary
        radexRead
        (
            const word& name,
            const Time& runTime,
            const dictionary& dict
        );


    //- Destructor
    virtual ~radexRead() = default;


    // Member Functions

        //- Read/re-read dictionary settings
        virtual bool read(const dictionary& dict);

        //- Fetch all configured fields from the key-value store
        virtual bool execute();

        //- No-op; field import happens on execute
        virtual bool write();
};


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace functionObjects
} // End namespace Foam

#endif
