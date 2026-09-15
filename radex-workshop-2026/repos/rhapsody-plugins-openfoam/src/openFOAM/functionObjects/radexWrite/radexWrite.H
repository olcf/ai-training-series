/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           |
     \\/     M anipulation  |
-------------------------------------------------------------------------------
Description
    Sends OpenFOAM fields to a radex key-value store at write intervals.

        The key naming convention is:
            <fieldName>_<timeStep>_<subdomainId>
        or (when identifier is configured):
            <identifier>_<fieldName>_<timeStep>_<subdomainId>

    Supported field types:
      - volScalarField             -> tensor of shape [nCells]  (via `fields`)
      - volVectorField             -> tensor of shape [nCells, 3] (via `fields`)
      - uniformDimensionedScalarField -> single scalar (via `scalars`)
      - Scalar results of another function object (e.g. surfaceFieldValue),
        looked up by that function object's name (via `scalars`). These are
        not objectRegistry objects; they are read from the function-object
        properties/results dictionary. If more than one result entry is
        registered under that name, only the first is exported.

        On finalization (`end()`), the final timestep is always written once
        by the master rank under:
            <identifier>_final_step
        If `identifier` is empty, the key is:
            final_step

    Backend is selected at runtime via the `backend` dictionary entry:
      backend   dragon;  (default)
      backend   redis;

    Example controlDict entry:
    \verbatim
    radexFields
    {
        type            radexWrite;
        libs            (radexWrite);

        fields          (U p nut nuTilda);
        scalars         (someAverage);
    }
    \endverbatim

\*---------------------------------------------------------------------------*/

#ifndef radexWrite_H
#define radexWrite_H

#include "radexBase.H"

#include "radex/handles.hpp"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{
namespace functionObjects
{

/*---------------------------------------------------------------------------*\
                         Class radexWrite Declaration
\*---------------------------------------------------------------------------*/

class radexWrite
:
    public radexBase
{
    // Private Member Functions

        //- Verb used in log messages
        virtual const char* actionName() const { return "Writing"; }

        //- Send a volScalarField to the key-value store
        virtual bool processScalarField
        (
            const word& fieldName,
            label subdomainId
        );

        //- Send a volVectorField to the key-value store
        virtual bool processVectorField
        (
            const word& fieldName,
            label subdomainId
        );

        //- Send a uniformDimensionedScalarField as a single scalar, or
        //- (if not found in the objectRegistry) the scalar result of
        //- another function object with that name
        virtual bool processUniformScalar
        (
            const word& fieldName,
            label subdomainId
        );

        //- Write final timestep metadata under <identifier>_final_step
        //- (master rank only)
        bool writeFinalStep();

        //- Build an OutgoingHandle from the shared key convention
        radex::data::OutgoingHandle
        makeHandle(const word& fieldName, label subdomainId) const;

        //- No copy construct
        radexWrite(const radexWrite&) = delete;

        //- No copy assignment
        void operator=(const radexWrite&) = delete;


public:

    //- Runtime type information
    TypeName("radexWrite");


    // Constructors

        //- Construct from Time and dictionary
        radexWrite
        (
            const word& name,
            const Time& runTime,
            const dictionary& dict
        );


    //- Destructor
    virtual ~radexWrite() = default;


    // Member Functions

        //- Execute (no-op; field export happens on write)
        virtual bool execute();

        //- Send all configured fields to the key-value store
        virtual bool write();

        //- Flush on simulation end
        virtual bool end();
};


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace functionObjects
} // End namespace Foam

#endif

// ************************************************************************* //
