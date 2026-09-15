/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           |
     \\/     M anipulation  |
-------------------------------------------------------------------------------
Description
    Common base for function objects that exchange OpenFOAM fields with a
    radex key-value store.

    Provides the pieces shared by radexRead and radexWrite:
      - the shared radex connection and key convention, via radexIO
      - the `fields` / `scalars` / `identifier` / `backend` dictionary entries
      - dispatch of each configured name to the matching typed handler

    Derived classes implement the typed handlers and decide which of the
    function object hooks (execute/write/end) triggers the transfer.

\*---------------------------------------------------------------------------*/

#ifndef radexBase_H
#define radexBase_H

#include "fvMeshFunctionObject.H"
#include "wordList.H"
#include "radexIO.H"

#include <string>

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{
namespace functionObjects
{

/*---------------------------------------------------------------------------*\
                         Class radexBase Declaration
\*---------------------------------------------------------------------------*/

class radexBase
:
    public fvMeshFunctionObject
{
protected:

    // Protected Data

        //- Names of volFields to exchange as tensors
        wordList fieldNames_;

        //- Names of scalars to exchange
        wordList scalarNames_;

        //- Shared radex connection and key convention
        radexIO io_;


    // Protected Member Functions

        //- Subdomain (MPI rank) owning this instance
        static label subdomainId();

        //- Build <identifier_>_<fieldName>_<timeStep>_<subdomainId>
        //- or <fieldName>_<timeStep>_<subdomainId>, and log it
        std::string makeKey(const word& fieldName, label subdomainId) const;

        //- Dispatch every configured name to its typed handler
        bool processFields();

        //- Verb used in log messages, e.g. "Writing"
        virtual const char* actionName() const = 0;

        //- Handle a name listed in `fields` that is a volScalarField
        virtual bool processScalarField
        (
            const word& fieldName,
            label subdomainId
        ) = 0;

        //- Handle a name listed in `fields` that is a volVectorField
        virtual bool processVectorField
        (
            const word& fieldName,
            label subdomainId
        ) = 0;

        //- Handle a name listed in `scalars`
        virtual bool processUniformScalar
        (
            const word& fieldName,
            label subdomainId
        ) = 0;

        //- No copy construct
        radexBase(const radexBase&) = delete;

        //- No copy assignment
        void operator=(const radexBase&) = delete;


public:

    //- Runtime type information
    TypeName("radexBase");


    // Constructors

        //- Construct from Time and dictionary; connects the radex client
        radexBase
        (
            const word& name,
            const Time& runTime,
            const dictionary& dict
        );


    //- Destructor
    virtual ~radexBase();


    // Member Functions

        //- Read the entries common to all radex function objects
        virtual bool read(const dictionary& dict);
};


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace functionObjects
} // End namespace Foam

#endif

// ************************************************************************* //
